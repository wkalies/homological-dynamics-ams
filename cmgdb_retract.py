
"""An order retraction of the SCCs of a CMGDB computation onto its Morse sets.

WHAT IT IS

Q is the poset of ALL strongly connected components of the multivalued map,
ordered by reachability; P is a sub-poset of it -- the Morse sets -- and
i : P -> Q is the inclusion.  An ORDER RETRACTION is a map sigma : Q ->> P that
is order preserving and fixes P:

    sigma(i(p)) = p          for every p in P
    q1 <= q2  =>  sigma(q1) <= sigma(q2)

It assigns every component of phase space, recurrent or not, to a Morse set
WITHOUT breaking the order.  Composed with "which component is this box in",
it labels every box with a Morse set, which is the decomposition of the domain
the Morse sets induce -- their basins.

Kalies, Kasti and VanderVorst, "Lattices and order in dynamics", give the
algorithm: Theorem 2.4 in general, and Corollary 2.5 for the case that covers
most runs, where every component outside P has exactly one Morse set
immediately below it and taking that one is already a retraction.

WHAT IT NEEDS

Only what CMGDB already returns:

    morse_graph, map_graph = CMGDB.ComputeConleyMorseGraph(model)
    r = cmgdb_retract.order_retraction(morse_graph, map_graph)
    print(r)
    cmgdb_retract.plot_basins(morse_graph, r, clist)

`map_graph` is the directed graph on boxes, and `morse_graph.morse_set(v)`
names the boxes of each Morse set in the same index space.  That is all the
algorithm is about -- it is order theory on a graph, and no geometry enters
until the picture.

WHEN IT EXISTS, AND WHEN IT DOES NOT

Not always.  A retraction exists exactly when no component's candidate set is
emptied, and this code assigns NOTHING unless a complete one exists: a partial
labeling would be read as a decomposition of the domain and it would not be
one.  `Retraction.status` says which case it is and `Retraction.ok` is the one
bit most callers want.

FORWARD INVARIANCE IS THE PRECONDITION.  If the domain is forward invariant
then every forward orbit stays inside it, so following the graph forward
always enters a cycle and every component reaches a Morse set.
Contrapositive: a component that reaches NO Morse set is one whose forward
orbit leaves the domain, which is what `status == NOT_FORWARD_INVARIANT`
reports.  On a domain that is not forward invariant the answer would be worse
than wrong, it would be vacuous -- the components that leave carry no
constraint, every Morse set is a candidate for each, and a "retraction" exists
that paints basin colors over a picture that means nothing.

HOW IT IS COMPUTED

Everything but one rare loop is numpy and scipy, so the work is linear in the
graph and done in C.

* The Morse sets reachable from each component come from ONE breadth-first
  search per Morse set in the transposed graph.  Every box of a Morse set lies
  in a single strongly connected component, so one source per set suffices,
  and a search is O(boxes + edges) in C rather than a sweep in Python.
* Each per-component question -- which reachable Morse sets are maximal, which
  Morse sets are candidates -- depends on the component only through one or
  two bitmasks, and millions of components carry a handful of distinct masks.
  So the Python work is per distinct mask and the rest is a lookup.
* The candidate machinery runs on the condensation, not on the box graph.

Needs numpy and scipy.  CMGDB is needed only for the type of the two
arguments; nothing here imports it, which also makes the module testable
against any other code that can produce a box graph and its Morse sets.
"""

import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

__all__ = ["Retraction", "order_retraction", "box_graph", "edge_components",
           "box_rectangles", "basin_rectangles", "phase_space_boxes",
           "morse_set_labels", "plot_labels", "plot_basins", "plot_morse_sets",
           "OK", "NOT_FORWARD_INVARIANT", "NO_RETRACTION", "BROKEN", "EMPTY"]

OK = 0                      #: sigma is complete
NOT_FORWARD_INVARIANT = 1   #: a component reaches no Morse set
NO_RETRACTION = 2           #: a candidate set emptied: no order retraction exists
BROKEN = 3                  #: one was built and failed its own check -- a bug
EMPTY = 4                   #: there are no Morse sets

_STATUS_TEXT = {
    OK: "ok",
    NOT_FORWARD_INVARIANT: ("a component reaches no Morse set, so its forward"
                            " orbit leaves the domain: the domain is not"
                            " forward invariant"),
    NO_RETRACTION: "a candidate set emptied: no order retraction exists",
    BROKEN: "a retraction was built and failed its own check -- a bug",
    EMPTY: "there are no Morse sets",
}


class Retraction(object):
    """What :func:`order_retraction` returns.

    sigma        the Morse set each BOX retracts onto, indexed as CMGDB
                 indexes boxes; empty unless the retraction is complete
    sigma_comp   the same, indexed by strongly connected component
    component    the component of each box
    status       OK, NOT_FORWARD_INVARIANT, NO_RETRACTION, BROKEN or EMPTY
    method       "corollary 2.5" or "candidates"
    components   how many strongly connected components there are
    transient    how many of them are not Morse sets
    chosen       of those, how many needed a choice between candidates
    rounds       trimming passes the candidate algorithm made
    offender     the component that refused, when one did
    seconds      wall clock, for the caller's own reporting
    """

    def __init__(self, **kw):
        self.sigma = kw.get("sigma", np.empty(0, np.int32))
        self.sigma_comp = kw.get("sigma_comp", np.empty(0, np.int32))
        self.component = kw.get("component", np.empty(0, np.int32))
        self.status = kw.get("status", BROKEN)
        self.method = kw.get("method", None)
        self.components = kw.get("components", 0)
        self.transient = kw.get("transient", 0)
        self.chosen = kw.get("chosen", 0)
        self.rounds = kw.get("rounds", 0)
        self.offender = kw.get("offender", -1)
        self.seconds = kw.get("seconds", 0.0)

    @property
    def ok(self):
        return self.status == OK

    def boxes_per_set(self, num_sets=None):
        """How many boxes retract onto each Morse set."""
        if not self.ok:
            return np.empty(0, np.int64)
        n = int(self.sigma.max()) + 1 if num_sets is None else num_sets
        return np.bincount(self.sigma, minlength=n).astype(np.int64)

    def __str__(self):
        if self.status != OK:
            s = "order retraction: NOT COMPUTED -- " + _STATUS_TEXT[self.status]
            if self.offender >= 0:
                s += " (component %d)" % self.offender
            return s
        s = ("order retraction: all %d components assigned to a Morse set by"
             " %s (%d outside P" % (self.components, self.method,
                                    self.transient))
        if self.chosen:
            s += ", %d needed a choice" % self.chosen
        if self.rounds:
            s += ", %d trimming passes" % self.rounds
        return s + "), order preserving on every edge -- checked"


# ---------------------------------------------------------------------------
# bit helpers.  A component's Morse sets are a bitmask in one int64, so there
# is a ceiling of 63 Morse sets, which is far above anything CMGDB produces.
# ---------------------------------------------------------------------------

_MAXSETS = 63


def _bits(mask, m):
    return [i for i in range(m) if (mask >> i) & 1]


def _popcount(a):
    out = np.zeros(a.shape, np.int32)
    for i in range(_MAXSETS):
        out += ((a >> np.int64(i)) & 1).astype(np.int32)
    return out


def _lowest_bit(a):
    """Index of the lowest set bit of each entry, -1 where there is none."""
    out = np.full(a.shape, -1, np.int32)
    for i in range(_MAXSETS - 1, -1, -1):
        out = np.where(((a >> np.int64(i)) & 1).astype(bool), np.int32(i), out)
    return out


def _per_distinct(values, fn):
    """`fn` applied to each DISTINCT value of `values`, mapped back."""
    uniq, inv = np.unique(values, return_inverse=True)
    table = np.array([fn(int(u)) for u in uniq], np.int64)
    return table[inv.reshape(values.shape)]


def _per_distinct2(a, b, fn):
    """The same for a question that depends on two masks.

    Two 63-bit masks do not fit in one int64, so the pair is made distinct by
    `np.unique` over the two columns rather than by packing them.
    """
    uniq, inv = np.unique(np.stack([a, b], 1), axis=0, return_inverse=True)
    table = np.array([fn(int(u), int(v)) for u, v in uniq], np.int64)
    return table[inv.reshape(a.shape)]


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------

def box_graph(map_graph, dtype=np.int32):
    """CMGDB's box graph as CSR arrays (indptr, indices).

    The only function here that touches CMGDB's API, and the only place the
    cost is Python's: one `adjacencies` call per box, which runs at something
    like 10**5 boxes a second.  Hand the result back to `order_retraction` as
    `graph=` if it is wanted twice.

    The edges go straight into a numpy buffer that grows geometrically,
    rather than into a list of arrays or a list of ints.  At the production
    scale that is not a detail: 46 million edges as Python ints are well over
    a gigabyte of object headers, and as eight million small arrays nearly as
    much again, while the buffer is 190 MB.
    """
    n = map_graph.num_vertices()
    indptr = np.zeros(n + 1, np.int64)
    buf = np.empty(max(1 << 16, 8 * n), dtype)
    k = 0
    for v in range(n):
        a = map_graph.adjacencies(v)
        m = len(a)
        if k + m > buf.size:
            bigger = np.empty(max(2 * buf.size, k + m), dtype)
            bigger[:k] = buf[:k]
            buf = bigger
        buf[k:k + m] = a
        k += m
        indptr[v + 1] = k
    return indptr, buf[:k].copy()


def _csr(indptr, indices, n):
    return csr_matrix((np.ones(indices.size, np.int8), indices, indptr),
                      shape=(n, n))


def edge_components(indptr, indices, labels, chunk=1 << 22):
    """The component of each end of each edge, a chunk of edges at a time,
    and only the edges that run BETWEEN components.

    At the production scale there are 10**8 edges, so materializing both ends
    as arrays -- let alone as int64 -- is a gigabyte that nothing needs: the
    two places that read the edges, the order check and the final
    verification, both read them once.
    """
    n = indptr.size - 1
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        a = np.repeat(labels[lo:hi], np.diff(indptr[lo:hi + 1]))
        b = labels[indices[indptr[lo]:indptr[hi]]]
        keep = a != b
        yield a[keep], b[keep]


def _reverse_topological(indptr, indices, labels, nc):
    """Relabel the components so every edge runs from a higher label to a
    lower one, which is what the sweeps below assume.

    scipy's strong components come out of Tarjan's algorithm and are already
    in that order, so this checks -- one pass over the edges, in chunks --
    and returns the labels untouched when they are.  The fallback is Kahn's
    algorithm, there so that nothing depends on an undocumented property of
    somebody else's implementation.
    """
    if all(bool(np.all(a > b)) for a, b in
           edge_components(indptr, indices, labels)):
        return labels, False

    parts = list(edge_components(indptr, indices, labels))
    ca = np.concatenate([p[0] for p in parts])
    cb = np.concatenate([p[1] for p in parts])
    del parts
    nxt = np.argsort(cb, kind="stable")
    cb_s, ca_s = cb[nxt], ca[nxt]
    start = np.searchsorted(cb_s, np.arange(nc))
    stop = np.searchsorted(cb_s, np.arange(nc), side="right")
    indeg = np.bincount(ca, minlength=nc).astype(np.int64)
    order = np.full(nc, -1, np.int64)
    ready = [int(c) for c in np.flatnonzero(indeg == 0)]
    seen = 0
    while ready:
        d = ready.pop()
        order[d] = seen
        seen += 1
        for c in ca_s[start[d]:stop[d]]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(int(c))
    if seen != nc:
        raise RuntimeError("the condensation has a cycle: not a condensation")
    return order[labels], True


# ---------------------------------------------------------------------------
# the retraction
# ---------------------------------------------------------------------------

def order_retraction(morse_graph, map_graph, graph=None,
                     verify_candidates=False, verbose=False):
    """The order retraction of a CMGDB computation onto its Morse sets.

    morse_graph, map_graph  as returned by CMGDB.ComputeConleyMorseGraph
    graph                   (indptr, indices) from `box_graph`, when it has
                            already been built -- that is the expensive part
    verify_candidates       run the general algorithm even where Corollary 2.5
                            applies, and check that the two agree.  It is the
                            strongest check there is on the candidate
                            machinery: where the corollary's hypothesis holds
                            its answer is a theorem, so the general algorithm
                            has to reproduce it component for component.
    """
    t0 = time.time()
    n = map_graph.num_vertices()
    M = morse_graph.num_vertices()
    if M == 0:
        return Retraction(status=EMPTY, seconds=time.time() - t0)
    if M > _MAXSETS:
        raise NotImplementedError("%d Morse sets: the bitmasks here hold %d"
                                  % (M, _MAXSETS))

    indptr, indices = graph if graph is not None else box_graph(map_graph)
    fwd = _csr(indptr, indices, n)

    nc, labels = connected_components(fwd, directed=True, connection="strong")
    labels = labels.astype(np.int32, copy=False)
    labels, relabeled = _reverse_topological(indptr, indices, labels, nc)
    if verbose:
        print("   %d boxes, %d edges, %d components%s"
              % (n, indices.size, nc, ", relabeled" if relabeled else ""))

    # own[c]: the Morse set that component c IS, or -1.  Every box of a Morse
    # set lies in one component, so one box of it names that component.
    own = np.full(nc, -1, np.int32)
    seed = np.empty(M, np.int64)
    for p in range(M):
        boxes = np.asarray(morse_graph.morse_set(p), dtype=np.int64)
        if boxes.size == 0:
            raise ValueError("Morse set %d is empty" % p)
        c = labels[boxes]
        if np.any(c != c[0]):
            raise ValueError("Morse set %d is not one strongly connected"
                             " component" % p)
        own[c[0]] = p
        seed[p] = boxes[0]
    is_own = own >= 0

    # DOWN: the Morse sets reachable from each component.  c reaches p exactly
    # when c is reachable from p in the TRANSPOSED graph, so one search per
    # Morse set answers it for every component at once.
    back = fwd.T.tocsr()
    down = np.zeros(nc, np.int64)
    for p in range(M):
        reach = breadth_first_order(back, seed[p], directed=True,
                                    return_predecessors=False)
        down[labels[reach]] |= np.int64(1) << np.int64(p)

    # The order on P, read off DOWN rather than off the drawn Morse graph, so
    # no convention of somebody else's picture can get it backwards.
    del back                       # 400 MB at the production scale
    reaches = np.array([int(down[labels[seed[p]]]) & ~(1 << p)
                        for p in range(M)], np.int64)
    bel = np.array([int(reaches[p]) | (1 << p) for p in range(M)], np.int64)
    abv = np.array([(1 << p) | sum((1 << k) for k in range(M)
                                   if k != p and (int(reaches[k]) >> p) & 1)
                    for p in range(M)], np.int64)

    # succ_P(q): the MAXIMAL Morse sets below q.  m is maximal among the ones
    # reachable from q when no other reachable k reaches m.
    def maximal(mask):
        out = 0
        for m in _bits(mask, M):
            if not any(k != m and (mask >> k) & 1
                       and (int(reaches[k]) >> m) & 1 for k in range(M)):
                out |= 1 << m
        return out

    succ = _per_distinct(down, maximal)
    count = _popcount(succ)
    transient = int(np.count_nonzero(~is_own))

    bad = np.flatnonzero((~is_own) & (count == 0))
    if bad.size:
        return Retraction(status=NOT_FORWARD_INVARIANT, components=nc,
                          transient=transient, offender=int(bad[0]),
                          component=labels.astype(np.int32),
                          seconds=time.time() - t0)

    all_unique = bool(np.all(count[~is_own] == 1))
    uniq = np.where(is_own, own, _lowest_bit(succ)).astype(np.int32)

    if all_unique and not verify_candidates:
        r = Retraction(sigma_comp=uniq, status=OK, method="corollary 2.5",
                       components=nc, transient=transient)
    else:
        r = _candidates(nc, M, own, is_own, down, labels, indptr, indices,
                        fwd, seed, reaches, bel, abv,
                        uniq if all_unique else None, transient, verbose)
    r.component = labels.astype(np.int32)
    r.components = nc
    if r.status != OK:
        r.seconds = time.time() - t0
        return r

    # VERIFY, always.  Two properties, neither expensive beside the sweeps
    # above: sigma fixes P and lands in P, and sigma is order preserving on
    # every edge of the condensation.  The second is what a wrong choice or a
    # mistaken trim would break, and it would break quietly -- the output is a
    # labeling either way, and a labeling that is not order preserving is not
    # a decomposition of the domain.
    s = r.sigma_comp
    if (np.any(s[is_own] != own[is_own]) or np.any(s < 0) or np.any(s >= M)):
        r.status = BROKEN
        r.seconds = time.time() - t0
        return r
    for ca, cb in edge_components(indptr, indices, labels):
        a, b = s[ca].astype(np.int64), s[cb].astype(np.int64)
        good = (a == b) | (((bel[a] >> b) & 1).astype(bool))
        if not np.all(good):
            r.status = BROKEN
            r.offender = int(ca[~good][0])
            r.seconds = time.time() - t0
            return r

    r.sigma = r.sigma_comp[labels].astype(np.int32)
    r.seconds = time.time() - t0
    return r


def _candidates(nc, M, own, is_own, down, labels, indptr, indices, fwd, seed,
                reaches, bel, abv, cross_check, transient, verbose):
    """Theorem 2.4: candidate sets, trimmed to a fixpoint, then a choice.

    Runs on the condensation rather than on the box graph, and on tables
    indexed by the distinct bitmasks rather than per component.
    """
    # UP: the Morse sets that reach each component -- DOWN's argument run
    # forwards.  Only the general path needs it, so only it pays for it.
    up = np.zeros(nc, np.int64)
    for p in range(M):
        reach = breadth_first_order(fwd, seed[p], directed=True,
                                    return_predecessors=False)
        up[labels[reach]] |= np.int64(1) << np.int64(p)

    # One edge per ordered pair of components; the trimming visits them many
    # times and the box graph has far more of them than the condensation does.
    key = np.unique(np.concatenate(
        [a.astype(np.int64) * nc + b for a, b in
         edge_components(indptr, indices, labels)]))
    ea, eb = key // np.int64(nc), key % np.int64(nc)
    o = np.argsort(ea, kind="stable")
    ea, eb = ea[o], eb[o]
    estart = np.searchsorted(ea, np.arange(nc))
    estop = np.searchsorted(ea, np.arange(nc), side="right")
    if verbose:
        print("   condensation: %d edges" % ea.size)

    # STEP 1.  cds[q] = { p : every Morse set below q is below p, and every
    # Morse set above q is above p }.  For q in P the candidate set is {q}:
    # sigma has to fix P, which is what makes it a retraction rather than a
    # labeling.
    def candidates_of(d, u):
        out = 0
        for p in range(M):
            if (d & ~int(bel[p])) == 0 and (u & ~int(abv[p])) == 0:
                out |= 1 << p
        return out
    cds = _per_distinct2(down, up, candidates_of)
    cds[is_own] = (np.int64(1) << own[is_own].astype(np.int64))

    # STEP 2.  Trim what no order-preserving map could use.  For an edge
    # q -> r of the condensation sigma(r) <= sigma(q), so p survives in cds[q]
    # only if some candidate of r lies at or below it, and symmetrically
    # upward.  Trimming over the direct edges rather than over the covers of Q
    # applies only real relations -- sound -- and the covers are among them,
    # so the fixpoint is the same; computing the covers of a poset with
    # millions of elements is the part that would not scale.
    def support_down(mask):      # the p that a candidate set `mask` can sit under
        return sum((1 << p) for p in range(M) if mask & int(bel[p]))

    def support_up(mask):        # the p that it can sit over
        return sum((1 << p) for p in range(M) if mask & int(abv[p]))

    rounds = 0
    while True:
        rounds += 1
        allow = np.full(nc, -1, np.int64)
        np.bitwise_and.at(allow, ea, _per_distinct(cds, support_down)[eb])
        new = np.where(is_own, cds, cds & allow)

        allow_up = np.full(nc, -1, np.int64)
        np.bitwise_and.at(allow_up, eb, _per_distinct(new, support_up)[ea])
        new = np.where(is_own, new, new & allow_up)

        empty = np.flatnonzero(new == 0)
        if empty.size:
            return Retraction(status=NO_RETRACTION, transient=transient,
                              rounds=rounds, offender=int(empty[0]))
        if np.array_equal(new, cds):
            break
        cds = new

    # STEP 3.  Choose -- in COMPONENT ORDER, and MINIMALLY.
    #
    # The paper chooses one value, re-trims everything and repeats; that is
    # right for a poset with a dozen elements and impossible with millions.
    # Every edge runs from a higher component index to a lower one, so walking
    # the components upward means each one's successors already have a value,
    # and choosing a candidate above all of them makes sigma order preserving
    # on those edges by construction rather than by luck -- the same
    # information the re-trim would have propagated, gathered in one pass.
    #
    # Among those, take a MINIMAL candidate: one with no other candidate
    # strictly below it.  Not the lowest numbered -- the numbering is Tarjan's
    # and means nothing here.  Minimal is also what makes this agree with
    # Corollary 2.5 exactly: when succ_P(q) is the single set m, every
    # candidate lies above m and m is itself a candidate, so m is the unique
    # minimum and the two paths cannot disagree.
    sigma = np.where(is_own, own, -1).astype(np.int32)
    forced = (~is_own) & (_popcount(cds) == 1)
    sigma[forced] = _lowest_bit(cds[forced])

    chosen = 0
    for c in np.flatnonzero((~is_own) & ~forced):
        ok = int(cds[c])
        for d in eb[estart[c]:estop[c]]:
            sd = int(sigma[d])
            if sd < 0:
                return Retraction(status=BROKEN, transient=transient,
                                  rounds=rounds, offender=int(c))
            ok &= int(abv[sd])        # the candidates at or above sigma(d)
        here = _bits(ok, M)
        if not here:
            return Retraction(status=BROKEN, transient=transient,
                              rounds=rounds, offender=int(c))
        minimal = [p for p in here
                   if not any(q != p and (int(reaches[p]) >> q) & 1
                              for q in here)]
        if len(minimal) > 1:
            chosen += 1
        sigma[c] = minimal[0]

    method = ("candidates (cross-checked against corollary 2.5)"
              if cross_check is not None else "candidates")
    if cross_check is not None and not np.array_equal(sigma, cross_check):
        where = np.flatnonzero(sigma != cross_check)
        return Retraction(status=BROKEN, transient=transient, rounds=rounds,
                          offender=int(where[0]), method=method)
    return Retraction(sigma_comp=sigma, status=OK, method=method,
                      transient=transient, chosen=chosen, rounds=rounds)


# ---------------------------------------------------------------------------
# the picture
# ---------------------------------------------------------------------------

def box_rectangles(labels, boxes, num_sets=None):
    """Boxes carrying the same label merged into rectangles.

    `labels` is one int per box, -1 for "draw nothing", and `boxes` the
    (n, 4) array of [x0, y0, x1, y1] that `phase_space_box` gives.  Returns
    one (k, 4) array of (x, y, width, height) per label.

    A basin is every box of the grid, and matplotlib will not draw eight
    million of them.  Boxes of the same color, row and size that touch are
    merged into runs first, which collapses a basin to a few thousand
    rectangles and keeps the figure vector rather than rasterized.

    Boxes are GROUPED on quantized coordinates and DRAWN from the original
    ones.  CMGDB's box corners come out of repeated bisection in floating
    point: at 2**16 the boxes of a single level carry nine distinct widths
    differing in the last bits, and the left edges drift by as much against a
    perfect lattice.  Comparing either exactly breaks a run every few boxes,
    which is not a small effect -- it costs a factor of twenty in the
    rectangle count.  The quantum is a thousandth of the smallest box, far
    below any real difference in box size, since levels differ by factors of
    two, and far above the noise.
    """
    sigma = np.asarray(labels)
    M = (int(sigma.max()) + 1) if num_sets is None else num_sets
    x0, y0 = boxes[:, 0], boxes[:, 1]
    w, h = boxes[:, 2] - x0, boxes[:, 3] - y0
    q = 1e-3 * min(w.min(), h.min())
    qx, qy = np.round(x0 / q), np.round(y0 / q)
    qw, qh = np.round(w / q), np.round(h / q)

    out = []
    for p in range(M):
        sel = np.flatnonzero(sigma == p)
        if sel.size == 0:
            out.append(np.empty((0, 4)))
            continue
        o = np.lexsort((qx[sel], qw[sel], qh[sel], qy[sel]))
        s = sel[o]
        b, ww, hh = qy[s], qw[s], qh[s]
        cut = np.ones(s.size, bool)
        if s.size > 1:
            # Same row, same size -- asked of the quantized values -- and
            # touching, which is asked of the ORIGINAL ones: quantizing a
            # position and a width rounds them independently, so the
            # quantized left edges of two touching boxes can differ by one
            # quantum and the run would break there for no reason.
            touch = np.abs(x0[s][1:] - (x0[s][:-1] + w[s][:-1]))
            cut[1:] = ~((b[1:] == b[:-1]) & (hh[1:] == hh[:-1])
                        & (ww[1:] == ww[:-1])
                        & (touch <= 1e-6 * w[s][:-1]))
        start = np.flatnonzero(cut)
        stop = np.append(start[1:], s.size) - 1
        head, tail = s[start], s[stop]
        out.append(np.stack([x0[head], y0[head],
                             x0[tail] + w[tail] - x0[head], h[head]], 1))
    return out


def phase_space_boxes(morse_graph, n):
    """The (n, 4) array of box corners, which both pictures want."""
    return np.array([morse_graph.phase_space_box(i) for i in range(n)], float)


def morse_set_labels(morse_graph, n):
    """One label per box: its Morse set, or -1 for the boxes in none."""
    lab = np.full(n, -1, np.int32)
    for p in range(morse_graph.num_vertices()):
        lab[np.asarray(morse_graph.morse_set(p), dtype=np.int64)] = p
    return lab


def basin_rectangles(morse_graph, retraction, num_sets=None, boxes=None):
    """`box_rectangles` for a retraction's basins."""
    if not retraction.ok:
        raise ValueError("no retraction to draw: " + str(retraction))
    if boxes is None:
        boxes = phase_space_boxes(morse_graph, retraction.sigma.size)
    if num_sets is None:
        num_sets = morse_graph.num_vertices()
    return box_rectangles(retraction.sigma, boxes, num_sets)


def plot_labels(labels, boxes, clist=None, num_sets=None, ax=None,
                figsize=(8, 6), dpi=150, lower_bounds=None, upper_bounds=None,
                title=None, fname=None, rasterized=True, save_dpi=600):
    """Draw labeled boxes, merged into rectangles, in `clist`'s colors.

    THE BOXES ARE RASTERIZED, the rest of the figure is not.  A basin is
    tiled by tens of thousands of abutting rectangles, and left as vector
    paths they are at the mercy of whatever composites them: this renderer
    fills them cleanly, and another may antialias both sides of every shared
    edge and leave a half-covered seam, which over a whole basin reads as a
    washed-out color.  One image at `save_dpi` cannot do that, and it is also
    what CMGDB's own Morse-set plots do, for the same reason and in the same
    words -- "stored as one 600-dpi image; the text, axes and Morse graph
    stay vector, so PDFs are small, fast in LaTeX, and sharp".

    Stroking each rectangle in its own fill color is the other obvious fix
    and it is a wrong one: measured through matplotlib's rasterizer, the
    unstroked fills come out exactly the listed color -- 0.00% of the pixels
    in a basin differ from it -- and a 0.3 pt stroke is what makes 74% of
    them differ, because the stroke itself is antialiased.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    labels = np.asarray(labels)
    M = (int(labels.max()) + 1) if num_sets is None else num_sets
    if clist is None:
        import CMGDB
        clist = CMGDB.DEFAULT_CLIST
    runs = box_rectangles(labels, boxes, M)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    for p in range(M):
        if runs[p].size == 0:
            continue
        ax.add_collection(PatchCollection(
            [Rectangle((x, y), a, b) for x, y, a, b in runs[p]],
            facecolor=clist[p % len(clist)], edgecolor="none", linewidth=0,
            rasterized=rasterized))
    if lower_bounds is not None and upper_bounds is not None:
        ax.set_xlim(lower_bounds[0], upper_bounds[0])
        ax.set_ylim(lower_bounds[1], upper_bounds[1])
    else:
        ax.autoscale_view()
    if title:
        ax.set_title(title, fontsize=10)
    if fname:
        fig.savefig(fname, bbox_inches="tight", dpi=save_dpi)
    return fig, ax


def plot_basins(morse_graph, retraction, clist=None, boxes=None, **kw):
    """Every box in the color of the Morse set it retracts onto."""
    if not retraction.ok:
        raise ValueError("no retraction to draw: " + str(retraction))
    if boxes is None:
        boxes = phase_space_boxes(morse_graph, retraction.sigma.size)
    return plot_labels(retraction.sigma, boxes, clist,
                       num_sets=morse_graph.num_vertices(), **kw)


def plot_morse_sets(morse_graph, n_boxes, clist=None, boxes=None, **kw):
    """The Morse sets themselves, drawn by the same code as the basins so the
    two pictures line up box for box."""
    if boxes is None:
        boxes = phase_space_boxes(morse_graph, n_boxes)
    return plot_labels(morse_set_labels(morse_graph, n_boxes), boxes, clist,
                       num_sets=morse_graph.num_vertices(), **kw)
