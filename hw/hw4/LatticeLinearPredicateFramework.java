package hw.hw4;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

// ============= Core Framework =============

public class LatticeLinearPredicateFramework {

    // Base global state interface
    public interface GlobalState {
        int size();
        Object snapshot();
    }

    // Integer array-based state (for most algorithms)
    public static class IntArrayState implements GlobalState {
        private final AtomicIntegerArray state;
        private final int n;

        public IntArrayState(int n) {
            this.n = n;
            this.state = new AtomicIntegerArray(n);
        }

        public int get(int index) {
            return state.get(index);
        }

        public void set(int index, int value) {
            state.set(index, value);
        }

        public boolean compareAndSet(int index, int expect, int update) {
            return state.compareAndSet(index, expect, update);
        }

        @Override
        public int size() {
            return n;
        }

        @Override
        public int[] snapshot() {
            int[] snapshot = new int[n];
            for (int i = 0; i < n; i++) {
                snapshot[i] = state.get(i);
            }
            return snapshot;
        }
    }

    // Double array-based state (for Bellman-Ford)
    public static class DoubleArrayState implements GlobalState {
        private final AtomicReferenceArray<Double> state;
        private final int n;

        public DoubleArrayState(int n) {
            this.n = n;
            this.state = new AtomicReferenceArray<>(n);
            for (int i = 0; i < n; i++) {
                state.set(i, Double.POSITIVE_INFINITY);
            }
        }

        public double get(int index) {
            return state.get(index);
        }

        public void set(int index, double value) {
            state.set(index, value);
        }

        public boolean compareAndSet(int index, double expect, double update) {
            return state.compareAndSet(index, expect, update);
        }

        @Override
        public int size() {
            return n;
        }

        @Override
        public double[] snapshot() {
            double[] snapshot = new double[n];
            for (int i = 0; i < n; i++) {
                snapshot[i] = state.get(i);
            }
            return snapshot;
        }
    }

    public static class ThreadContext {
        public final int j;
        public final GlobalState G;

        public ThreadContext(int j, GlobalState G) {
            this.j = j;
            this.G = G;
        }
    }

    public interface LLPAlgorithm {
        GlobalState createGlobalState(int n);
        void init(ThreadContext ctx);
        boolean forbidden(ThreadContext ctx);
        void advance(ThreadContext ctx);
        default boolean hasConverged(GlobalState G) {
            return false;
        }
    }

    public static class LLPRunner {
        private final int n;
        private final int numPlatformThreads;
        private final GlobalState G;
        private final LLPAlgorithm algorithm;
        private final ExecutorService executor;
        private final AtomicBoolean running;
        private final AtomicInteger activeThreads;
        private final CountDownLatch initLatch;

        public LLPRunner(int n, LLPAlgorithm algorithm) {
            this(n, algorithm, Runtime.getRuntime().availableProcessors());
        }

        public LLPRunner(int n, LLPAlgorithm algorithm, int numPlatformThreads) {
            this.n = n;
            this.numPlatformThreads = Math.min(numPlatformThreads, n);
            this.G = algorithm.createGlobalState(n);
            this.algorithm = algorithm;
            this.executor = Executors.newFixedThreadPool(this.numPlatformThreads);
            this.running = new AtomicBoolean(true);
            this.activeThreads = new AtomicInteger(0);
            this.initLatch = new CountDownLatch(n);
        }

        public void start() {
            for (int j = 0; j < n; j++) {
                final int threadID = j;
                ThreadContext ctx = new ThreadContext(threadID, G);
                algorithm.init(ctx);
                initLatch.countDown();
            }

            for (int p = 0; p < numPlatformThreads; p++) {
                final int platformThreadId = p;
                executor.submit(() -> platformThreadLoop(platformThreadId));
            }
        }

        public void stop() {
            running.set(false);
            executor.shutdown();
        }

        public boolean awaitConvergence(long timeout, TimeUnit unit) throws InterruptedException {
            long deadline = System.nanoTime() + unit.toNanos(timeout);
            while (running.get() && System.nanoTime() < deadline) {
                if (algorithm.hasConverged(G)) {
                    stop();
                    return true;
                }
                Thread.sleep(10);
            }
            return algorithm.hasConverged(G);
        }

        public GlobalState getGlobalState() {
            return G;
        }

        private void platformThreadLoop(int platformThreadId) {
            activeThreads.incrementAndGet();

            try {
                initLatch.await();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                activeThreads.decrementAndGet();
                return;
            }

            while (running.get()) {
                for (int j = platformThreadId; j < n; j += numPlatformThreads) {
                    ThreadContext ctx = new ThreadContext(j, G);

                    if (algorithm.forbidden(ctx)) {
                        algorithm.advance(ctx);
                    }
                }
                //Thread.yield();
            }
            activeThreads.decrementAndGet();
        }
    }
}

// ============= Algorithm Implementations =============

// Has to be outside of LatticeLinearPredicateFramework otherwise get some obscure error.
// I don't know, I'm not an OOP guy haha
// Wow does Java have a metric ton of ways to create "this thing must implmenet these methods"
// This class is motivated by the potential edge case of hasConverged returning true when it should 
// return false due to a value being mutated while hasConverged is scanning G.
abstract class ConvergenceCheckerLLPAlgorithm implements LatticeLinearPredicateFramework.LLPAlgorithm {
    private static final int CONVERGENCE_CHECK_COUNT = 5;
    private static final int CONVERGENCE_CHECK_DELAY_MS = 5;

    /**
     * Checks if the algorithm has converged on a single snapshot of the global state.
     * This is the core algorithm-specific convergence predicate.
     * 
     * @param G The global state.
     * @return true if converged, false otherwise.
     */
    //protected abstract boolean isLocallyConverged(LatticeLinearPredicateFramework.GlobalState G);
    protected boolean isLocallyConverged(LatticeLinearPredicateFramework.GlobalState G) {
        for (int j = 0; j < G.size(); j++) {
            LatticeLinearPredicateFramework.ThreadContext ctx = 
                new LatticeLinearPredicateFramework.ThreadContext(j, G);
            if (forbidden(ctx)) {
                return false; // Found a forbidden state
            }
        }
        return true; // No forbidden states found
    }

    @Override
    public boolean hasConverged(LatticeLinearPredicateFramework.GlobalState G) {
        // Check multiple times over a short period to avoid issue of global state
        // potentially being mutated while it is being scanned
        
        for (int i = 0; i < CONVERGENCE_CHECK_COUNT; i++) {
            if (!isLocallyConverged(G)) {
                return false;
            }
            try {
                // Wait a short time to allow other threads to potentially advance/mutate the state
                Thread.sleep(CONVERGENCE_CHECK_DELAY_MS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
        }
        return true;
    }
}

// Initially done as implements LLPAlgorithm, but need for multiple convergence checks
// Prompeted switching to terribly named ConvergenceCheckerLLPAlgorithm
// Does technically leave room for edge cases.
// Using a data structure that supports lock-free modification, but has the ability to
// be frozen might be useful, but this solution is decent enough.
class StableMarriageAlgorithm extends ConvergenceCheckerLLPAlgorithm {
    private final int[][] mpref;
    private final int[][] rank;
    private final int[][] I;

    public StableMarriageAlgorithm(int[][] mpref, int[][] rank, int[][] I) {
        this.mpref = mpref;
        this.rank = rank;
        this.I = I;
    }

    @Override
    public LatticeLinearPredicateFramework.GlobalState createGlobalState(int n) {
        return new LatticeLinearPredicateFramework.IntArrayState(n);
    }

    @Override
    public void init(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState state = 
            (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        state.set(ctx.j, I[ctx.j][0]);
    }

    @Override
    public boolean forbidden(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState state = 
            (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        
        int j = ctx.j;
        int Gj = state.get(j);
        
        if (Gj >= mpref[j].length) return false;
        
        int z = mpref[j][Gj];
        
        for (int i = 0; i < state.size(); i++) {
            int Gi = state.get(i);
            for (int k = 0; k <= Gi && k < mpref[i].length; k++) {
                if (z == mpref[i][k] && rank[z][i] < rank[z][j]) {
                    return true;
                }
            }
        }
        return false;
    }

    @Override
    public void advance(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState state = 
            (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        state.set(ctx.j, state.get(ctx.j) + 1);
    }
}

class ScanAlgorithm extends ConvergenceCheckerLLPAlgorithm {
    private final int[] A;
    private final int[] S;

    public ScanAlgorithm(int[] A) {
        this.A = A;
        int n = A.length;
        this.S = new int[n];
        
        // Build S safely - only compute for valid indices
        for (int i = 0; i < n; i++) {
            int leftChild = 2 * i + 1;
            int rightChild = 2 * i + 2;
            S[i] = (leftChild < n ? A[leftChild] : 0) + 
                   (rightChild < n ? A[rightChild] : 0);
        }
    }

    @Override
    public LatticeLinearPredicateFramework.GlobalState createGlobalState(int n) {
        return new LatticeLinearPredicateFramework.IntArrayState(2 * n - 1);
    }

    @Override
    public void init(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState state = 
            (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        state.set(ctx.j, Integer.MIN_VALUE);
    }

    @Override
    public boolean forbidden(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState state = 
            (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        int j = ctx.j;
        int Gj = state.get(j);
        int n = A.length;

        // Check bounds before every array access
        if (j == 1 && Gj < 0) return true;
        
        if (j % 2 == 0) {
            int parentIndex = j / 2;
            if (parentIndex < state.size() && Gj < state.get(parentIndex)) 
                return true;
        }
        
        if (j % 2 == 1) {
            int parentIndex = j / 2;
            if (parentIndex < state.size()) {
                if (j < n && j - 1 < S.length && Gj < S[j - 1] + state.get(parentIndex)) 
                    return true;
                if (j >= n && j - n < A.length && Gj < A[j - n] + state.get(parentIndex)) 
                    return true;
            }
        }

        return false;
    }

    @Override
    public void advance(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState state = 
            (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        int j = ctx.j;
        int n = A.length;
        int newValue = Integer.MIN_VALUE;

        if (j == 1) newValue = Math.max(newValue, 0);
        
        if (j % 2 == 0) {
            int parentIndex = j / 2;
            if (parentIndex < state.size()) {
                newValue = Math.max(newValue, state.get(parentIndex));
            }
        }
        
        if (j % 2 == 1) {
            int parentIndex = j / 2;
            if (parentIndex < state.size()) {
                if (j < n && j - 1 < S.length) {
                    newValue = Math.max(newValue, S[j - 1] + state.get(parentIndex));
                }
                if (j >= n && j - n < A.length) {
                    newValue = Math.max(newValue, A[j - n] + state.get(parentIndex));
                }
            }
        }

        state.set(j, newValue);
    }

    @Override
    protected boolean isLocallyConverged(LatticeLinearPredicateFramework.GlobalState G) {
        for (int j = 0; j < G.size(); j++) {
            LatticeLinearPredicateFramework.ThreadContext ctx = 
                new LatticeLinearPredicateFramework.ThreadContext(j, G);
            if (forbidden(ctx)) {
                return false; // Found a forbidden state
            }
        }
        return true; // No forbidden states found
    }
}


class LLPReduce extends ConvergenceCheckerLLPAlgorithm {
    private final int[] A;
    private final int n_A; // This is 'n' from the pseudo-code

    public LLPReduce(int[] A) {
        this.A = A;
        this.n_A = A.length;
    }

    @Override
    public LatticeLinearPredicateFramework.GlobalState createGlobalState(int n) {
        // n here is the size of the state G, which is n_A - 1
        return new LatticeLinearPredicateFramework.IntArrayState(n);
    }

    @Override
    public void init(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState G = (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        G.set(ctx.j, Integer.MIN_VALUE); // -inf
    }

    @Override
    public boolean forbidden(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState G = (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        int j = ctx.j;       // 0-based Java index
        int j_p = j + 1;   // 1-based pseudo-code index
        int g_j = G.get(j);

        // Clause 2: G[j] >= A[2j - n + 1] + A[2j - n + 2] if n/2 <= j < n
        // These are the "leaf-parents" in the reduction tree.
        if (j_p >= n_A / 2) {
            // Convert pseudo-code A[k] to Java A[k-1]
            int a_idx_1 = (2 * j_p - n_A + 1) - 1; // 2*j_p - n_A
            int a_idx_2 = (2 * j_p - n_A + 2) - 1; // 2*j_p - n_A + 1
            
            // Check for potential overflow before adding
            long sum = (long)A[a_idx_1] + A[a_idx_2];
            if (sum > Integer.MAX_VALUE) {
                 return g_j < Integer.MAX_VALUE; // or handle overflow appropriately
            }
            if (sum < Integer.MIN_VALUE) {
                return g_j < Integer.MIN_VALUE;
            }

            return g_j < (int)sum;
        }
        // Clause 1: G[j] >= G[2j] + G[2j + 1] if 1 <= j < n/2
        // These are the internal nodes.
        else {
            // Convert pseudo-code G[k] to Java G[k-1]
            int left_child_j = (2 * j_p) - 1;
            int right_child_j = (2 * j_p + 1) - 1; // 2 * j_p

            int g_left = G.get(left_child_j);
            int g_right = G.get(right_child_j);

            // Dependency check: wait for children to be computed
            if (g_left == Integer.MIN_VALUE || g_right == Integer.MIN_VALUE) {
                return false; // Not forbidden, dependencies not met
            }
            
            // Check for potential overflow
            long sum = (long)g_left + g_right;
            if (sum > Integer.MAX_VALUE) {
                return g_j < Integer.MAX_VALUE;
            }
            if (sum < Integer.MIN_VALUE) {
                return g_j < Integer.MIN_VALUE;
            }

            return g_j < (int)sum;
        }
    }

    @Override
    public void advance(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState G = (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        int j = ctx.j;
        int j_p = j + 1;

        // Clause 2 (Leaf-parents)
        if (j_p >= n_A / 2) {
            int a_idx_1 = 2 * j_p - n_A;
            int a_idx_2 = 2 * j_p - n_A + 1;
            
            long sum = (long)A[a_idx_1] + A[a_idx_2];
            int val = (int)Math.max(Integer.MIN_VALUE, Math.min(Integer.MAX_VALUE, sum));
            G.set(j, val);
        }
        // Clause 1 (Internal nodes)
        else {
            int left_child_j = (2 * j_p) - 1;
            int right_child_j = (2 * j_p);

            int g_left = G.get(left_child_j);
            int g_right = G.get(right_child_j);
            
            // This advance should only be called if forbidden was true,
            // meaning children were not MIN_VALUE.
            long sum = (long)g_left + g_right;
            int val = (int)Math.max(Integer.MIN_VALUE, Math.min(Integer.MAX_VALUE, sum));
            G.set(j, val);
        }
    }
}

class LLPScan extends ConvergenceCheckerLLPAlgorithm {
    private final int[] A;
    private final int[] S; // Summation tree from LLPReduce
    private final int n_A; // This is 'n' from the pseudo-code

    /**
     * @param A The original input array (size n)
     * @param S The summation tree from LLPReduce (size n-1)
     */
    public LLPScan(int[] A, int[] S) {
        this.A = A;
        this.S = S;
        this.n_A = A.length;
    }

    @Override
    public LatticeLinearPredicateFramework.GlobalState createGlobalState(int n) {
        // n here is the size of state G, which is 2*n_A - 1
        return new LatticeLinearPredicateFramework.IntArrayState(n);
    }

    @Override
    public void init(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState G = (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        G.set(ctx.j, Integer.MIN_VALUE); // -inf
    }

    @Override
    public boolean forbidden(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState G = (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        int j = ctx.j;       // 0-based Java index
        int j_p = j + 1;   // 1-based pseudo-code index
        int g_j = G.get(j);

        // Clause 1: G[j] >= 0 if j = 1
        if (j == 0) { // j_p == 1
            return g_j < 0;
        }

        // Parent pseudo-index: j_p / 2
        // Parent Java index: (j_p / 2) - 1  ==>  ((j + 1) / 2) - 1
        int parent_j = ((j + 1) / 2) - 1;
        int g_parent = G.get(parent_j);

        if (g_parent == Integer.MIN_VALUE) {
            return false;
        }

        // Clause 2: G[j] >= G[j/2] if j is even (Left child)
        if (j_p % 2 == 0) { // j_p is even, so j is odd
            return g_j < g_parent;
        }
        
        // j_p must be odd (j is even)
        
        // Clause 3: G[j] >= S[j - 1] + G[j/2] if j is odd and j < n
        if (j_p < n_A) { // Internal right child
            // S_pseudo[k] -> S_java[k-1]
            // We need S_pseudo[j_p - 1]
            int s_idx = (j_p - 1) - 1; // j - 1
            
            // S array from LLPReduce might be shorter if n=1
            if (s_idx < 0 || s_idx >= S.length) {
                 // This case should ideally not be hit if j_p > 1 and j_p < n_A (implies n_A > 2)
                 // But as a safeguard:
                 return g_j < g_parent; // S[j-1] is effectively 0
            }
            
            int s_val = S[s_idx];
            
            long sum = (long)s_val + g_parent;
            if (sum > Integer.MAX_VALUE) return g_j < Integer.MAX_VALUE;
            if (sum < Integer.MIN_VALUE) return g_j < Integer.MIN_VALUE;

            return g_j < (int)sum;
        }
        // Clause 4: G[j] >= A[j - n] + G[j/2] if j is odd and j > n
        else { // j_p >= n_A (Leaf node). Since j_p is odd, j_p > n_A (as n_A is power of 2)
               // Note: pseudo-code says j > n, which is correct as n is even.
            
            // A_pseudo[k] -> A_java[k-1]
            // We need A_pseudo[j_p - n_A]
            int a_idx = (j_p - n_A) - 1; // (j+1 - n_A) - 1 = j - n_A
            int a_val = A[a_idx];
            
            long sum = (long)a_val + g_parent;
            if (sum > Integer.MAX_VALUE) return g_j < Integer.MAX_VALUE;
            if (sum < Integer.MIN_VALUE) return g_j < Integer.MIN_VALUE;

            return g_j < (int)sum;
        }
    }

    @Override
    public void advance(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState G = (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        int j = ctx.j;
        int j_p = j + 1;

        // Clause 1
        if (j == 0) {
            G.set(0, 0);
            return;
        }

        // Parent value (must be computed if forbidden was true)
        int parent_j = ((j + 1) / 2) - 1;
        int g_parent = G.get(parent_j);

        // Clause 2 (Left child)
        if (j_p % 2 == 0) {
            G.set(j, g_parent);
            return;
        }

        // j_p must be odd (j is even)

        // Clause 3 (Internal right child)
        if (j_p < n_A) {
            int s_idx = j - 1; // (j_p - 1) - 1
            int s_val = S[s_idx];
            
            long sum = (long)s_val + g_parent;
            int val = (int)Math.max(Integer.MIN_VALUE, Math.min(Integer.MAX_VALUE, sum));
            G.set(j, val);
        }
        // Clause 4 (Leaf right child)
        else {
            int a_idx = j - n_A; // (j_p - n_A) - 1
            int a_val = A[a_idx];
            
            long sum = (long)a_val + g_parent;
            int val = (int)Math.max(Integer.MIN_VALUE, Math.min(Integer.MAX_VALUE, sum));
            G.set(j, val);
        }
    }
}

class FastComponentsAlgorithm extends ConvergenceCheckerLLPAlgorithm {
    private final List<Integer>[] adj;

    @SuppressWarnings("unchecked")
    public FastComponentsAlgorithm(List<Integer>[] adj) {
        this.adj = adj;
    }

    @Override
    public LatticeLinearPredicateFramework.GlobalState createGlobalState(int n) {
        return new LatticeLinearPredicateFramework.IntArrayState(n);
    }

    @Override
    public void init(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState state = 
            (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        state.set(ctx.j, ctx.j);
    }

    @Override
    public boolean forbidden(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState state = 
            (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        int j = ctx.j;
        int parentJ = state.get(j);
        int parentParentJ = state.get(parentJ);

        if (parentJ < parentParentJ) return true;

        for (int i : adj[j]) {
            if (parentJ < state.get(i)) return true;
        }

        return false;
    }

    @Override
    public void advance(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState state = 
            (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        int j = ctx.j;
        int parentJ = state.get(j);
        int newValue = state.get(parentJ);

        for (int i : adj[j]) {
            newValue = Math.max(newValue, state.get(i));
        }

        state.set(j, newValue);
    }
}

class BellmanFordAlgorithm extends ConvergenceCheckerLLPAlgorithm {
    private final List<Integer>[] pre;
    private final double[][] w;
    private final int source;

    public BellmanFordAlgorithm(List<Integer>[] pre, double[][] w, int source) {
        this.pre = pre;
        this.w = w;
        this.source = source;
    }

    @Override
    public LatticeLinearPredicateFramework.GlobalState createGlobalState(int n) {
        return new LatticeLinearPredicateFramework.DoubleArrayState(n);
    }

    @Override
    public void init(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.DoubleArrayState state = 
            (LatticeLinearPredicateFramework.DoubleArrayState) ctx.G;
        state.set(ctx.j, ctx.j == source ? 0.0 : Double.POSITIVE_INFINITY);
    }

    @Override
    public boolean forbidden(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.DoubleArrayState state = 
            (LatticeLinearPredicateFramework.DoubleArrayState) ctx.G;
        int j = ctx.j;
        double dj = state.get(j);

        for (int i : pre[j]) {
            if (dj > state.get(i) + w[i][j]) {
                return true;
            }
        }
        return false;
    }

    @Override
    public void advance(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.DoubleArrayState state = 
            (LatticeLinearPredicateFramework.DoubleArrayState) ctx.G;
        int j = ctx.j;
        double minDist = Double.POSITIVE_INFINITY;

        for (int i : pre[j]) {
            minDist = Math.min(minDist, state.get(i) + w[i][j]);
        }

        state.set(j, minDist);
    }
}

class JohnsonAlgorithm extends ConvergenceCheckerLLPAlgorithm {//implements LatticeLinearPredicateFramework.LLPAlgorithm {
    private final List<Integer>[] pre;
    private final int[][] w;
    private int tempMax;

    public JohnsonAlgorithm(List<Integer>[] pre, int[][] w) {
        this.pre = pre;
        this.w = w;
        this.tempMax = Integer.MIN_VALUE;
    }

    @Override
    public LatticeLinearPredicateFramework.GlobalState createGlobalState(int n) {
        return new LatticeLinearPredicateFramework.IntArrayState(n);
    }

    @Override
    public void init(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState state = 
            (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        state.set(ctx.j, 0);
    }

    @Override
    public boolean forbidden(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState state = 
            (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        int j = ctx.j;
        int pj = state.get(j);

        tempMax = Integer.MIN_VALUE;

        // Calculate the max here and not in advance otherwise risk issues with 
        // values being mutated
        for (int i : pre[j]) {
            tempMax = Math.max(tempMax, pj - w[i][j]);
        }

        if (pj < tempMax) {
            return true;
        }
        return false;
    }

    @Override
    public void advance(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState state = 
            (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        int j = ctx.j;

        state.set(j, tempMax);
    }
}

class BoruvkaAlgorithm extends ConvergenceCheckerLLPAlgorithm {
    public static class Edge {
        public final int u;
        public final int v;
        public final double weight;
        
        public Edge(int u, int v, double weight) {
            this.u = u;
            this.v = v;
            this.weight = weight;
        }
        
        @Override
        public String toString() {
            return "(" + u + "," + v + "," + weight + ")";
        }
    }
    
    private final List<Edge>[] adjList;
    private final int n;
    private final Set<Edge> msfEdges;
    
    @SuppressWarnings("unchecked")
    public BoruvkaAlgorithm(List<Edge>[] adjList, int n) {
        this.adjList = adjList;
        this.n = n;
        this.msfEdges = ConcurrentHashMap.newKeySet();
    }
    
    @Override
    public LatticeLinearPredicateFramework.GlobalState createGlobalState(int n) {
        return new LatticeLinearPredicateFramework.IntArrayState(n);
    }
    
    // Find minimum weight edge from vertex v
    private Edge findMinWeightEdge(int v, LatticeLinearPredicateFramework.IntArrayState state) {
        Edge minEdge = null;
        double minWeight = Double.POSITIVE_INFINITY;
        
        for (Edge e : adjList[v]) {
            int neighbor = (e.u == v) ? e.v : e.u;
            // Only consider edges to different components
            if (state.get(v) != state.get(neighbor) && e.weight < minWeight) {
                minWeight = e.weight;
                minEdge = e;
            }
        }
        
        return minEdge;
    }
    
    @Override
    public void init(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState state = 
            (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        int v = ctx.j;
        
        // Find minimum weight edge from v
        Edge mweV = findMinWeightEdge(v, state);
        
        if (mweV == null) {
            // No outgoing edges, v is its own parent
            state.set(v, v);
            return;
        }
        
        int w = (mweV.u == v) ? mweV.v : mweV.u;
        
        // Find minimum weight edge from w
        Edge mweW = findMinWeightEdge(w, state);
        
        int parent = 0;
        if (mweW != null && edgesEqual(mweV, mweW)) {
            // Break symmetry: if both vertices choose each other, lower index becomes parent
            parent = w;
        } else if (mweV.equals(mweW) && v < w) {
            parent = v;
        } else {
            parent = w;
        }
        
        state.set(v, parent);
        
        // Add edge to MSF (avoid duplicates by ensuring u < v)
        int edgeU = Math.min(v, w);
        int edgeV = Math.max(v, w);
        msfEdges.add(new Edge(edgeU, edgeV, mweV.weight));
    }
    
    private boolean edgesEqual(Edge e1, Edge e2) {
        if (e1 == null || e2 == null) return false;
        return (e1.u == e2.u && e1.v == e2.v) || (e1.u == e2.v && e1.v == e2.u);
    }
    
    @Override
    public boolean forbidden(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState state = 
            (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        int j = ctx.j;
        int Gj = state.get(j);
        int GGj = state.get(Gj);
        
        // forbidden(j) ≡ G[j] != G[G[j]]
        return Gj != GGj;
    }
    
    @Override
    public void advance(LatticeLinearPredicateFramework.ThreadContext ctx) {
        LatticeLinearPredicateFramework.IntArrayState state = 
            (LatticeLinearPredicateFramework.IntArrayState) ctx.G;
        int j = ctx.j;
        
        // advance(j) def G[j] := G[G[j]]
        state.set(j, state.get(state.get(j)));
    }
    
    public Set<Edge> getMSFEdges() {
        return new HashSet<>(msfEdges);
    }
    
    // Verify MSF correctness
    public boolean verifyMSF() {
        // Check connectivity within components
        LatticeLinearPredicateFramework.IntArrayState finalState = 
            (LatticeLinearPredicateFramework.IntArrayState) createGlobalState(n);
        
        // Simple verification: check that we have n - numComponents edges
        // where numComponents is the number of distinct roots
        Set<Integer> roots = new HashSet<>();
        for (int i = 0; i < n; i++) {
            int root = i;
            Set<Integer> visited = new HashSet<>();
            while (root != finalState.get(root) && !visited.contains(root)) {
                visited.add(root);
                root = finalState.get(root);
            }
            roots.add(root);
        }
        
        int expectedEdges = n - roots.size();
        return msfEdges.size() <= expectedEdges;
    }
}

// ============= Test Case Generators =============
// Beware! Lots of AI generated code below.

class TestCaseGenerator {
    private final Random random;

    public TestCaseGenerator(long seed) {
        this.random = new Random(seed);
    }

    // Generate random stable marriage instance
    public static class StableMarriageInstance {
        public final int[][] mpref;
        public final int[][] rank;
        public final int[][] I;
        public final int n;

        public StableMarriageInstance(int[][] mpref, int[][] rank, int[][] I, int n) {
            this.mpref = mpref;
            this.rank = rank;
            this.I = I;
            this.n = n;
        }
    }

    public StableMarriageInstance generateStableMarriage(int n) {
        int[][] mpref = new int[n][n];
        int[][] wpref = new int[n][n];
        int[][] rank = new int[n][n];
        int[][] I = new int[n][1];

        // Generate random preferences for men
        for (int i = 0; i < n; i++) {
            List<Integer> women = new ArrayList<>();
            for (int j = 0; j < n; j++) women.add(j);
            Collections.shuffle(women, random);
            for (int j = 0; j < n; j++) {
                mpref[i][j] = women.get(j);
            }
            I[i][0] = 0; // Start at first preference
        }

        // Generate random preferences for women
        for (int i = 0; i < n; i++) {
            List<Integer> men = new ArrayList<>();
            for (int j = 0; j < n; j++) men.add(j);
            Collections.shuffle(men, random);
            for (int j = 0; j < n; j++) {
                wpref[i][j] = men.get(j);
            }
        }

        // Build rank matrix (inverse of preference)
        for (int woman = 0; woman < n; woman++) {
            for (int j = 0; j < n; j++) {
                int man = wpref[woman][j];
                rank[woman][man] = j;
            }
        }

        return new StableMarriageInstance(mpref, rank, I, n);
    }

    // Generate scan input
    public int[] generateScanInput(int n) {
        int[] A = new int[n];
        for (int i = 0; i < n; i++) {
            A[i] = random.nextInt(100);
        }
        return A;
    }

    // Generate random graph for connected components
    @SuppressWarnings("unchecked")
    public List<Integer>[] generateRandomGraph(int n, double edgeProbability) {
        List<Integer>[] adj = new ArrayList[n];
        for (int i = 0; i < n; i++) {
            adj[i] = new ArrayList<>();
        }

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (random.nextDouble() < edgeProbability) {
                    adj[i].add(j);
                    adj[j].add(i);
                }
            }
        }
        return adj;
    }

    // Generate graph for Bellman-Ford (DAG or general graph)
    public static class GraphInstance {
        public final List<Integer>[] adj;
        public final double[][] weights;
        public final int n;

        public GraphInstance(List<Integer>[] adj, double[][] weights, int n) {
            this.adj = adj;
            this.weights = weights;
            this.n = n;
        }
    }

    @SuppressWarnings("unchecked")
    public GraphInstance generateBellmanFordGraph(int n, int avgDegree) {
        List<Integer>[] adj = new ArrayList[n];
        double[][] weights = new double[n][n];
        
        for (int i = 0; i < n; i++) {
            adj[i] = new ArrayList<>();
        }

        // Create connected graph with approximately avgDegree edges per node
        for (int i = 0; i < n; i++) {
            int numEdges = Math.max(1, avgDegree + random.nextInt(3) - 1);
            for (int k = 0; k < numEdges && adj[i].size() < n - 1; k++) {
                int j = random.nextInt(n);
                if (j != i && !adj[i].contains(j)) {
                    adj[i].add(j);
                    weights[i][j] = 1 + random.nextDouble() * 10;
                }
            }
        }

        return new GraphInstance(adj, weights, n);
    }

    @SuppressWarnings("unchecked")
    public GraphInstance generateJohnsonGraph(int n, int avgDegree) {
        List<Integer>[] adj = new ArrayList[n];
        int[][] weights = new int[n][n];
        
        for (int i = 0; i < n; i++) {
            adj[i] = new ArrayList<>();
        }

        for (int i = 0; i < n; i++) {
            int numEdges = Math.max(1, avgDegree + random.nextInt(3) - 1);
            for (int k = 0; k < numEdges && adj[i].size() < n - 1; k++) {
                int j = random.nextInt(n);
                if (j != i && !adj[i].contains(j)) {
                    adj[i].add(j);
                    weights[i][j] = random.nextInt(20) - 5;
                }
            }
        }

        // Convert to predecessor lists for algorithms
        List<Integer>[] pre = new ArrayList[n];
        for (int i = 0; i < n; i++) {
            pre[i] = new ArrayList<>();
        }
        
        for (int i = 0; i < n; i++) {
            for (int j : adj[i]) {
                pre[j].add(i);
            }
        }

        return new GraphInstance(pre, convertToDouble(weights), n);
    }

    // Generate weighted undirected graph for Boruvka's algorithm
    @SuppressWarnings("unchecked")
    public List<BoruvkaAlgorithm.Edge>[] generateBoruvkaGraph(int n, double edgeProbability) {
        List<BoruvkaAlgorithm.Edge>[] adjList = new ArrayList[n];
        for (int i = 0; i < n; i++) {
            adjList[i] = new ArrayList<>();
        }
        
        // Generate random edges
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (random.nextDouble() < edgeProbability) {
                    double weight = 1 + random.nextDouble() * 99; // weights between 1 and 100
                    BoruvkaAlgorithm.Edge edge = new BoruvkaAlgorithm.Edge(i, j, weight);
                    adjList[i].add(edge);
                    adjList[j].add(edge);
                }
            }
        }
        
        return adjList;
    }
    
    // Generate connected weighted graph for Boruvka
    @SuppressWarnings("unchecked")
    public List<BoruvkaAlgorithm.Edge>[] generateConnectedBoruvkaGraph(int n) {
        List<BoruvkaAlgorithm.Edge>[] adjList = new ArrayList[n];
        for (int i = 0; i < n; i++) {
            adjList[i] = new ArrayList<>();
        }
        
        // First create a spanning tree to ensure connectivity
        List<Integer> inTree = new ArrayList<>();
        List<Integer> notInTree = new ArrayList<>();
        
        inTree.add(0);
        for (int i = 1; i < n; i++) {
            notInTree.add(i);
        }
        
        // Add edges to create spanning tree
        while (!notInTree.isEmpty()) {
            int treeVertex = inTree.get(random.nextInt(inTree.size()));
            int newVertex = notInTree.remove(random.nextInt(notInTree.size()));
            
            double weight = 1 + random.nextDouble() * 99;
            BoruvkaAlgorithm.Edge edge = new BoruvkaAlgorithm.Edge(treeVertex, newVertex, weight);
            adjList[treeVertex].add(edge);
            adjList[newVertex].add(edge);
            
            inTree.add(newVertex);
        }
        
        // Add some additional random edges
        int additionalEdges = n / 2;
        for (int k = 0; k < additionalEdges; k++) {
            int i = random.nextInt(n);
            int j = random.nextInt(n);
            
            if (i != j) {
                // Check if edge already exists
                boolean exists = false;
                for (BoruvkaAlgorithm.Edge e : adjList[i]) {
                    if ((e.u == i && e.v == j) || (e.u == j && e.v == i)) {
                        exists = true;
                        break;
                    }
                }
                
                if (!exists) {
                    double weight = 1 + random.nextDouble() * 99;
                    BoruvkaAlgorithm.Edge edge = new BoruvkaAlgorithm.Edge(i, j, weight);
                    adjList[i].add(edge);
                    adjList[j].add(edge);
                }
            }
        }
        
        return adjList;
    }

    private double[][] convertToDouble(int[][] arr) {
        double[][] result = new double[arr.length][arr.length];
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr.length; j++) {
                result[i][j] = arr[i][j];
            }
        }
        return result;
    }
}

// ============= Benchmarking Framework =============

class LLPBenchmark {
    
    public static class BenchmarkResult {
        public final String algorithmName;
        public final int problemSize;
        public final int numThreads;
        public final long executionTimeMs;
        public final boolean converged;
        public final String inputFile;
        public final String outputFile;

        public BenchmarkResult(String algorithmName, int problemSize, int numThreads, 
                             long executionTimeMs, boolean converged, String inputFile, String outputFile) {
            this.algorithmName = algorithmName;
            this.problemSize = problemSize;
            this.numThreads = numThreads;
            this.executionTimeMs = executionTimeMs;
            this.converged = converged;
            this.inputFile = inputFile;
            this.outputFile = outputFile;
        }

        @Override
        public String toString() {
            return String.format("%s | n=%d | threads=%d | time=%dms | converged=%b",
                algorithmName, problemSize, numThreads, executionTimeMs, converged);
        }
    }

    private final TestCaseGenerator generator;
    private final List<BenchmarkResult> results;
    private final String outputDir;
    private int runCounter;

    public LLPBenchmark(long seed) {
        this(seed, "benchmark_output");
    }

    public LLPBenchmark(long seed, String outputDir) {
        this.generator = new TestCaseGenerator(seed);
        this.results = new ArrayList<>();
        this.outputDir = outputDir;
        this.runCounter = 0;
        
        // Create output directory
        new java.io.File(outputDir).mkdirs();
    }

    public void runStableMarriageBenchmark(int n, int numThreads, long timeoutSeconds) {
        System.out.println("\n=== Benchmarking Stable Marriage (n=" + n + ", threads=" + numThreads + ") ===");
        
        String runId = String.format("stablemarriage_n%d_t%d_r%d", n, numThreads, runCounter++);
        String inputFile = outputDir + "/" + runId + "_input.txt";
        String outputFile = outputDir + "/" + runId + "_output.txt";
        
        TestCaseGenerator.StableMarriageInstance instance = generator.generateStableMarriage(n);
        
        // Write input to file
        try (java.io.PrintWriter writer = new java.io.PrintWriter(inputFile)) {
            writer.println("Stable Marriage Problem");
            writer.println("n = " + n);
            writer.println("\nMen's Preferences:");
            for (int i = 0; i < n; i++) {
                writer.print("Man " + i + ": ");
                for (int j = 0; j < n; j++) {
                    writer.print(instance.mpref[i][j] + " ");
                }
                writer.println();
            }
            writer.println("\nWomen's Rank Matrix:");
            for (int i = 0; i < n; i++) {
                writer.print("Woman " + i + ": ");
                for (int j = 0; j < n; j++) {
                    writer.print(instance.rank[i][j] + " ");
                }
                writer.println();
            }
        } catch (Exception e) {
            System.err.println("Error writing input file: " + e.getMessage());
        }
        
        StableMarriageAlgorithm algo = new StableMarriageAlgorithm(
            instance.mpref, instance.rank, instance.I);
        
        long startTime = System.currentTimeMillis();
        LatticeLinearPredicateFramework.LLPRunner runner = 
            new LatticeLinearPredicateFramework.LLPRunner(n, algo, numThreads);
        runner.start();
        
        boolean converged = false;
        try {
            converged = runner.awaitConvergence(timeoutSeconds, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        runner.stop();
        long executionTime = System.currentTimeMillis() - startTime;
        
        // Write output to file
        try (java.io.PrintWriter writer = new java.io.PrintWriter(outputFile)) {
            writer.println("Stable Marriage Results");
            writer.println("Execution Time: " + executionTime + " ms");
            writer.println("Converged: " + converged);
            writer.println("Threads: " + numThreads);
            writer.println("\nFinal Matching (Man -> Woman):");
            
            LatticeLinearPredicateFramework.IntArrayState state = 
                (LatticeLinearPredicateFramework.IntArrayState) runner.getGlobalState();
            int[] finalState = state.snapshot();
            for (int i = 0; i < n; i++) {
                writer.println("Man " + i + " -> Woman " + finalState[i]);
            }
        } catch (Exception e) {
            System.err.println("Error writing output file: " + e.getMessage());
        }
        
        BenchmarkResult result = new BenchmarkResult(
            "StableMarriage", n, numThreads, executionTime, converged, inputFile, outputFile);
        results.add(result);
        System.out.println(result);
    }

    public void runScanBenchmark(int n, int numThreads, long timeoutSeconds) {
        System.out.println("\n=== Benchmarking Scan (n=" + n + ", threads=" + numThreads + ") ===");
        
        String runId = String.format("scan_n%d_t%d_r%d", n, numThreads, runCounter++);
        String inputFile = outputDir + "/" + runId + "_input.txt";
        String outputFile = outputDir + "/" + runId + "_output.txt";
        
        int[] A = generator.generateScanInput(n);
        
        // Write input to file
        try (java.io.PrintWriter writer = new java.io.PrintWriter(inputFile)) {
            writer.println("Scan Problem");
            writer.println("n = " + n);
            writer.println("\nInput Array:");
            for (int i = 0; i < n; i++) {
                writer.print(A[i] + " ");
                if ((i + 1) % 20 == 0) writer.println();
            }
            writer.println();
        } catch (Exception e) {
            System.err.println("Error writing input file: " + e.getMessage());
        }
        
        ScanAlgorithm algo = new ScanAlgorithm(A);
        
        long startTime = System.currentTimeMillis();
        LatticeLinearPredicateFramework.LLPRunner runner = 
            new LatticeLinearPredicateFramework.LLPRunner(2 * n - 1, algo, numThreads);
        runner.start();
        
        boolean converged = false;
        try {
            converged = runner.awaitConvergence(timeoutSeconds, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        runner.stop();
        long executionTime = System.currentTimeMillis() - startTime;
        
        // Write output to file
        try (java.io.PrintWriter writer = new java.io.PrintWriter(outputFile)) {
            writer.println("Scan Results");
            writer.println("Execution Time: " + executionTime + " ms");
            writer.println("Converged: " + converged);
            writer.println("Threads: " + numThreads);
            writer.println("\nPrefix Sums:");
            
            LatticeLinearPredicateFramework.IntArrayState state = 
                (LatticeLinearPredicateFramework.IntArrayState) runner.getGlobalState();
            int[] finalState = state.snapshot();
            for (int i = n; i < 2 * n - 1; i++) {
                writer.println("Position " + (i - n) + ": " + finalState[i]);
            }
        } catch (Exception e) {
            System.err.println("Error writing output file: " + e.getMessage());
        }
        
        BenchmarkResult result = new BenchmarkResult(
            "Scan", n, numThreads, executionTime, converged, inputFile, outputFile);
        results.add(result);
        System.out.println(result);
    }

    public void runComponentsBenchmark(int n, double edgeProb, int numThreads, long timeoutSeconds) {
        System.out.println("\n=== Benchmarking Connected Components (n=" + n + ", threads=" + numThreads + ") ===");
        
        String runId = String.format("components_n%d_t%d_r%d", n, numThreads, runCounter++);
        String inputFile = outputDir + "/" + runId + "_input.txt";
        String outputFile = outputDir + "/" + runId + "_output.txt";
        
        List<Integer>[] adj = generator.generateRandomGraph(n, edgeProb);
        
        // Write input to file
        try (java.io.PrintWriter writer = new java.io.PrintWriter(inputFile)) {
            writer.println("Connected Components Problem");
            writer.println("n = " + n);
            writer.println("Edge Probability = " + edgeProb);
            writer.println("\nAdjacency Lists:");
            int edgeCount = 0;
            for (int i = 0; i < n; i++) {
                writer.print("Vertex " + i + ": ");
                for (int neighbor : adj[i]) {
                    writer.print(neighbor + " ");
                    if (neighbor > i) edgeCount++;
                }
                writer.println();
            }
            writer.println("Total Edges: " + edgeCount);
        } catch (Exception e) {
            System.err.println("Error writing input file: " + e.getMessage());
        }
        
        FastComponentsAlgorithm algo = new FastComponentsAlgorithm(adj);
        
        long startTime = System.currentTimeMillis();
        LatticeLinearPredicateFramework.LLPRunner runner = 
            new LatticeLinearPredicateFramework.LLPRunner(n, algo, numThreads);
        runner.start();
        
        boolean converged = false;
        try {
            converged = runner.awaitConvergence(timeoutSeconds, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        runner.stop();
        long executionTime = System.currentTimeMillis() - startTime;
        
        // Write output to file
        try (java.io.PrintWriter writer = new java.io.PrintWriter(outputFile)) {
            writer.println("Connected Components Results");
            writer.println("Execution Time: " + executionTime + " ms");
            writer.println("Converged: " + converged);
            writer.println("Threads: " + numThreads);
            writer.println("\nComponent Assignment:");
            
            LatticeLinearPredicateFramework.IntArrayState state = 
                (LatticeLinearPredicateFramework.IntArrayState) runner.getGlobalState();
            int[] finalState = state.snapshot();
            
            Map<Integer, List<Integer>> components = new HashMap<>();
            for (int i = 0; i < n; i++) {
                int root = finalState[i];
                components.computeIfAbsent(root, k -> new ArrayList<>()).add(i);
            }
            
            writer.println("Number of Components: " + components.size());
            int compId = 0;
            for (Map.Entry<Integer, List<Integer>> entry : components.entrySet()) {
                writer.print("Component " + compId++ + " (root=" + entry.getKey() + "): ");
                for (int v : entry.getValue()) {
                    writer.print(v + " ");
                }
                writer.println();
            }
        } catch (Exception e) {
            System.err.println("Error writing output file: " + e.getMessage());
        }
        
        BenchmarkResult result = new BenchmarkResult(
            "FastComponents", n, numThreads, executionTime, converged, inputFile, outputFile);
        results.add(result);
        System.out.println(result);
    }

    public void runBellmanFordBenchmark(int n, int avgDegree, int numThreads, long timeoutSeconds) {
        System.out.println("\n=== Benchmarking Bellman-Ford (n=" + n + ", threads=" + numThreads + ") ===");
        
        String runId = String.format("bellmanford_n%d_t%d_r%d", n, numThreads, runCounter++);
        String inputFile = outputDir + "/" + runId + "_input.txt";
        String outputFile = outputDir + "/" + runId + "_output.txt";
        
        TestCaseGenerator.GraphInstance graph = generator.generateBellmanFordGraph(n, avgDegree);
        
        // Write input to file
        try (java.io.PrintWriter writer = new java.io.PrintWriter(inputFile)) {
            writer.println("Bellman-Ford Problem");
            writer.println("n = " + n);
            writer.println("Source = 0");
            writer.println("\nEdges (u -> v, weight):");
            int edgeCount = 0;
            for (int i = 0; i < n; i++) {
                for (int j : graph.adj[i]) {
                    writer.printf("%d -> %d, weight: %.2f%n", i, j, graph.weights[i][j]);
                    edgeCount++;
                }
            }
            writer.println("Total Edges: " + edgeCount);
        } catch (Exception e) {
            System.err.println("Error writing input file: " + e.getMessage());
        }
        
        // Convert to predecessor lists
        @SuppressWarnings("unchecked")
        List<Integer>[] pre = new ArrayList[n];
        for (int i = 0; i < n; i++) {
            pre[i] = new ArrayList<>();
        }
        for (int i = 0; i < n; i++) {
            for (int j : graph.adj[i]) {
                pre[j].add(i);
            }
        }
        
        BellmanFordAlgorithm algo = new BellmanFordAlgorithm(pre, graph.weights, 0);
        
        long startTime = System.currentTimeMillis();
        LatticeLinearPredicateFramework.LLPRunner runner = 
            new LatticeLinearPredicateFramework.LLPRunner(n, algo, numThreads);
        runner.start();
        
        boolean converged = false;
        try {
            converged = runner.awaitConvergence(timeoutSeconds, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        runner.stop();
        long executionTime = System.currentTimeMillis() - startTime;
        
        // Write output to file
        try (java.io.PrintWriter writer = new java.io.PrintWriter(outputFile)) {
            writer.println("Bellman-Ford Results");
            writer.println("Execution Time: " + executionTime + " ms");
            writer.println("Converged: " + converged);
            writer.println("Threads: " + numThreads);
            writer.println("\nShortest Distances from Source 0:");
            
            LatticeLinearPredicateFramework.DoubleArrayState state = 
                (LatticeLinearPredicateFramework.DoubleArrayState) runner.getGlobalState();
            double[] finalState = state.snapshot();
            for (int i = 0; i < n; i++) {
                writer.printf("Vertex %d: %.2f%n", i, finalState[i]);
            }
        } catch (Exception e) {
            System.err.println("Error writing output file: " + e.getMessage());
        }
        
        BenchmarkResult result = new BenchmarkResult(
            "BellmanFord", n, numThreads, executionTime, converged, inputFile, outputFile);
        results.add(result);
        System.out.println(result);
    }

    public void runJohnsonBenchmark(int n, int avgDegree, int numThreads, long timeoutSeconds) {
        System.out.println("\n=== Benchmarking Johnson (n=" + n + ", threads=" + numThreads + ") ===");
        
        String runId = String.format("johnson_n%d_t%d_r%d", n, numThreads, runCounter++);
        String inputFile = outputDir + "/" + runId + "_input.txt";
        String outputFile = outputDir + "/" + runId + "_output.txt";
        
        TestCaseGenerator.GraphInstance graph = generator.generateJohnsonGraph(n, avgDegree);
        
        // Write input to file
        try (java.io.PrintWriter writer = new java.io.PrintWriter(inputFile)) {
            writer.println("Johnson Problem");
            writer.println("n = " + n);
            writer.println("\nPredecessor Lists and Weights:");
            
            @SuppressWarnings("unchecked")
            List<Integer>[] pre = (List<Integer>[]) graph.adj;
            int edgeCount = 0;
            for (int j = 0; j < n; j++) {
                writer.println("Vertex " + j + " predecessors:");
                for (int i : pre[j]) {
                    writer.printf("  %d -> %d, weight: %.0f%n", i, j, graph.weights[i][j]);
                    edgeCount++;
                }
            }
            writer.println("Total Edges: " + edgeCount);
        } catch (Exception e) {
            System.err.println("Error writing input file: " + e.getMessage());
        }
        
        // Graph already has predecessor format from generator
        @SuppressWarnings("unchecked")
        List<Integer>[] pre = (List<Integer>[]) graph.adj;
        
        // Convert weights to int
        int[][] intWeights = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                intWeights[i][j] = (int) graph.weights[i][j];
            }
        }
        
        JohnsonAlgorithm algo = new JohnsonAlgorithm(pre, intWeights);
        
        long startTime = System.currentTimeMillis();
        LatticeLinearPredicateFramework.LLPRunner runner = 
            new LatticeLinearPredicateFramework.LLPRunner(n, algo, numThreads);
        runner.start();
        
        boolean converged = false;
        try {
            converged = runner.awaitConvergence(timeoutSeconds, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        runner.stop();
        long executionTime = System.currentTimeMillis() - startTime;
        
        // Write output to file
        try (java.io.PrintWriter writer = new java.io.PrintWriter(outputFile)) {
            writer.println("Johnson Results");
            writer.println("Execution Time: " + executionTime + " ms");
            writer.println("Converged: " + converged);
            writer.println("Threads: " + numThreads);
            writer.println("\nMinimum Price Vector:");
            
            LatticeLinearPredicateFramework.IntArrayState state = 
                (LatticeLinearPredicateFramework.IntArrayState) runner.getGlobalState();
            int[] finalState = state.snapshot();
            for (int i = 0; i < n; i++) {
                writer.println("Vertex " + i + ": " + finalState[i]);
            }
        } catch (Exception e) {
            System.err.println("Error writing output file: " + e.getMessage());
        }
        
        BenchmarkResult result = new BenchmarkResult(
            "Johnson", n, numThreads, executionTime, converged, inputFile, outputFile);
        results.add(result);
        System.out.println(result);
    }

    public void runBoruvkaBenchmark(int n, double edgeProbability, int numThreads, long timeoutSeconds) {
        System.out.println("\n=== Benchmarking Boruvka MSF (n=" + n + ", threads=" + numThreads + ") ===");
        
        String runId = String.format("boruvka_n%d_t%d_r%d", n, numThreads, runCounter++);
        String inputFile = outputDir + "/" + runId + "_input.txt";
        String outputFile = outputDir + "/" + runId + "_output.txt";
        
        List<BoruvkaAlgorithm.Edge>[] adjList = generator.generateConnectedBoruvkaGraph(n);
        
        // Write input to file
        try (java.io.PrintWriter writer = new java.io.PrintWriter(inputFile)) {
            writer.println("Boruvka MSF Problem");
            writer.println("n = " + n);
            writer.println("\nEdges:");
            Set<String> printedEdges = new HashSet<>();
            int edgeCount = 0;
            for (int i = 0; i < n; i++) {
                for (BoruvkaAlgorithm.Edge e : adjList[i]) {
                    String edgeKey = Math.min(e.u, e.v) + "-" + Math.max(e.u, e.v);
                    if (!printedEdges.contains(edgeKey)) {
                        writer.printf("(%d, %d) weight: %.2f%n", e.u, e.v, e.weight);
                        printedEdges.add(edgeKey);
                        edgeCount++;
                    }
                }
            }
            writer.println("Total Edges: " + edgeCount);
        } catch (Exception e) {
            System.err.println("Error writing input file: " + e.getMessage());
        }
        
        BoruvkaAlgorithm algo = new BoruvkaAlgorithm(adjList, n);
        
        long startTime = System.currentTimeMillis();
        LatticeLinearPredicateFramework.LLPRunner runner = 
            new LatticeLinearPredicateFramework.LLPRunner(n, algo, numThreads);
        runner.start();
        
        boolean converged = false;
        try {
            converged = runner.awaitConvergence(timeoutSeconds, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        runner.stop();
        long executionTime = System.currentTimeMillis() - startTime;
        
        // Get MSF edges
        Set<BoruvkaAlgorithm.Edge> msfEdges = algo.getMSFEdges();
        
        // Write output to file
        try (java.io.PrintWriter writer = new java.io.PrintWriter(outputFile)) {
            writer.println("Boruvka MSF Results");
            writer.println("Execution Time: " + executionTime + " ms");
            writer.println("Converged: " + converged);
            writer.println("Threads: " + numThreads);
            writer.println("\nMinimum Spanning Forest:");
            writer.println("Number of edges: " + msfEdges.size());
            
            double totalWeight = 0;
            for (BoruvkaAlgorithm.Edge e : msfEdges) {
                writer.printf("(%d, %d) weight: %.2f%n", e.u, e.v, e.weight);
                totalWeight += e.weight;
            }
            writer.printf("Total MSF Weight: %.2f%n", totalWeight);
            
            // Component information
            writer.println("\nFinal Component Structure:");
            LatticeLinearPredicateFramework.IntArrayState state = 
                (LatticeLinearPredicateFramework.IntArrayState) runner.getGlobalState();
            int[] finalState = state.snapshot();
            
            Map<Integer, List<Integer>> components = new HashMap<>();
            for (int i = 0; i < n; i++) {
                int root = finalState[i];
                components.computeIfAbsent(root, k -> new ArrayList<>()).add(i);
            }
            
            writer.println("Number of Components: " + components.size());
            int compId = 0;
            for (Map.Entry<Integer, List<Integer>> entry : components.entrySet()) {
                writer.print("Component " + compId++ + " (root=" + entry.getKey() + "): ");
                for (int v : entry.getValue()) {
                    writer.print(v + " ");
                }
                writer.println();
            }
        } catch (Exception e) {
            System.err.println("Error writing output file: " + e.getMessage());
        }
        
        System.out.println("MSF has " + msfEdges.size() + " edges");
        
        BenchmarkResult result = new BenchmarkResult(
            "Boruvka", n, numThreads, executionTime, converged, inputFile, outputFile);
        results.add(result);
        System.out.println(result);
    }




    public void runComprehensiveBenchmark() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("COMPREHENSIVE LLP ALGORITHM BENCHMARK");
        System.out.println("=".repeat(80));

        int[] sizes = {10, 50, 100, 500, 10000};
        int[] threadCounts = {1, 2, 4, 8};
        long timeout = 60; // seconds

        // Stable Marriage Benchmarks
        //System.out.println("\n" + "=".repeat(80));
        //System.out.println("STABLE MARRIAGE BENCHMARKS");
        //System.out.println("=".repeat(80));
        //for (int size : sizes) {
        //    for (int threads : threadCounts) {
        //        if (threads <= size) {
        //            runStableMarriageBenchmark(size, threads, timeout);
        //        }
        //    }
        //}

        // Scan Benchmarks (use powers of 2)
        System.out.println("\n" + "=".repeat(80));
        System.out.println("SCAN BENCHMARKS");
        System.out.println("=".repeat(80));
        int[] scanSizes = {16, 64, 1024, 4096, 8192};
        for (int size : scanSizes) {
            for (int threads : threadCounts) {
                runScanBenchmark(size, threads, timeout);
            }
        }

        // Connected Components Benchmarks
        //System.out.println("\n" + "=".repeat(80));
        //System.out.println("CONNECTED COMPONENTS BENCHMARKS");
        //System.out.println("=".repeat(80));
        //for (int size : sizes) {
        //    for (int threads : threadCounts) {
        //        if (threads <= size) {
        //            runComponentsBenchmark(size, 0.1, threads, timeout);
        //        }
        //    }
        //}

        // Bellman-Ford Benchmarks
        //System.out.println("\n" + "=".repeat(80));
        //System.out.println("BELLMAN-FORD BENCHMARKS");
        //System.out.println("=".repeat(80));
        //for (int size : sizes) {
        //    for (int threads : threadCounts) {
        //        if (threads <= size) {
        //            runBellmanFordBenchmark(size, 4, threads, timeout);
        //        }
        //    }
        //}

        // Johnson Benchmarks
        //System.out.println("\n" + "=".repeat(80));
        //System.out.println("JOHNSON BENCHMARKS");
        //System.out.println("=".repeat(80));
        //for (int size : sizes) {
        //    for (int threads : threadCounts) {
        //        if (threads <= size) {
        //            runJohnsonBenchmark(size, 4, threads, timeout);
        //        }
        //    }
        //}

        // Boruvka Benchmarks
        //System.out.println("\n" + "=".repeat(80));
        //System.out.println("BORUVKA MSF BENCHMARKS");
        //System.out.println("=".repeat(80));
        //for (int size : sizes) {
        //    for (int threads : threadCounts) {
        //        if (threads <= size) {
        //            runBoruvkaBenchmark(size, 0.2, threads, timeout);
        //        }
        //    }
        //}

        printSummary();
    }

    public void printSummary() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("BENCHMARK SUMMARY");
        System.out.println("=".repeat(80));
        
        // Group by algorithm
        Map<String, List<BenchmarkResult>> byAlgorithm = new HashMap<>();
        for (BenchmarkResult result : results) {
            byAlgorithm.computeIfAbsent(result.algorithmName, k -> new ArrayList<>()).add(result);
        }

        for (String algo : byAlgorithm.keySet()) {
            System.out.println("\n" + algo + ":");
            System.out.println("-".repeat(80));
            System.out.printf("%-10s %-10s %-15s %-12s%n", "Size", "Threads", "Time (ms)", "Converged");
            System.out.println("-".repeat(80));
            
            List<BenchmarkResult> algoResults = byAlgorithm.get(algo);
            algoResults.sort(Comparator.comparingInt((BenchmarkResult r) -> r.problemSize)
                                       .thenComparingInt(r -> r.numThreads));
            
            for (BenchmarkResult result : algoResults) {
                System.out.printf("%-10d %-10d %-15d %-12s%n",
                    result.problemSize, result.numThreads, 
                    result.executionTimeMs, result.converged);
            }
        }

        // Speedup analysis
        System.out.println("\n" + "=".repeat(80));
        System.out.println("SPEEDUP ANALYSIS (vs single thread)");
        System.out.println("=".repeat(80));
        
        for (String algo : byAlgorithm.keySet()) {
            System.out.println("\n" + algo + ":");
            List<BenchmarkResult> algoResults = byAlgorithm.get(algo);
            
            Map<Integer, Long> baselineTimes = new HashMap<>();
            for (BenchmarkResult result : algoResults) {
                if (result.numThreads == 1) {
                    baselineTimes.put(result.problemSize, result.executionTimeMs);
                }
            }
            
            System.out.printf("%-10s %-10s %-15s %-12s%n", "Size", "Threads", "Time (ms)", "Speedup");
            System.out.println("-".repeat(80));
            
            for (BenchmarkResult result : algoResults) {
                Long baseTime = baselineTimes.get(result.problemSize);
                if (baseTime != null && result.executionTimeMs > 0) {
                    double speedup = (double) baseTime / result.executionTimeMs;
                    System.out.printf("%-10d %-10d %-15d %.2fx%n",
                        result.problemSize, result.numThreads, 
                        result.executionTimeMs, speedup);
                }
            }
        }
    }

    public List<BenchmarkResult> getResults() {
        return new ArrayList<>(results);
    }

    // Export results to CSV
    public void exportToCSV(String filename) {
        try (java.io.PrintWriter writer = new java.io.PrintWriter(filename)) {
            writer.println("Algorithm,ProblemSize,NumThreads,ExecutionTimeMs,Converged");
            for (BenchmarkResult result : results) {
                writer.printf("%s,%d,%d,%d,%b%n",
                    result.algorithmName, result.problemSize, result.numThreads,
                    result.executionTimeMs, result.converged);
            }
            System.out.println("\nResults exported to " + filename);
        } catch (Exception e) {
            System.err.println("Error exporting to CSV: " + e.getMessage());
        }
    }
}

// ============= Main Class for Running Benchmarks =============

class LLPBenchmarkRunner {
    public static void main(String[] args) {
        System.out.println("Starting LLP Algorithm Benchmarks...");
        System.out.println("Available processors: " + Runtime.getRuntime().availableProcessors());
        
        LLPBenchmark benchmark = new LLPBenchmark(42); // Fixed seed for reproducibility
        
        // Run comprehensive benchmark
        benchmark.runComprehensiveBenchmark();
        
        // Export results
        benchmark.exportToCSV("llp_benchmark_results.csv");
        
        System.out.println("\n" + "=".repeat(80));
        System.out.println("BENCHMARKING COMPLETE");
        System.out.println("=".repeat(80));
    }
    
    // Example: Run individual benchmarks with custom parameters
    public static void customBenchmark() {
        LLPBenchmark benchmark = new LLPBenchmark(System.currentTimeMillis());
        
        // Custom Stable Marriage test
        benchmark.runStableMarriageBenchmark(100, 4, 30);
        
        // Custom Scan test with large input
        benchmark.runScanBenchmark(2048, 8, 30);
        
        // Custom Components test with dense graph
        benchmark.runComponentsBenchmark(200, 0.3, 4, 30);
        
        // Custom Bellman-Ford with many edges
        benchmark.runBellmanFordBenchmark(100, 8, 4, 30);
        
        // Custom Johnson test
        benchmark.runJohnsonBenchmark(100, 6, 4, 60);
        
        benchmark.printSummary();
        benchmark.exportToCSV("custom_benchmark_results.csv");
    }
}