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
                Thread.yield();
            }
            activeThreads.decrementAndGet();
        }
    }
}

// ============= Algorithm Implementations =============

class StableMarriageAlgorithm implements LatticeLinearPredicateFramework.LLPAlgorithm {
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

class ScanAlgorithm implements LatticeLinearPredicateFramework.LLPAlgorithm {
    private final int[] A;
    private final int[] S;

    public ScanAlgorithm(int[] A) {
        this.A = A;
        int n = A.length;
        this.S = new int[n - 1];
        for (int i = 0; i < n - 1; i++) {
            S[i] = A[2 * i + 1] + (2 * i + 2 < n ? A[2 * i + 2] : 0);
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

        if (j == 1 && Gj < 0) return true;
        if (j % 2 == 0 && j / 2 < state.size()) {
            if (Gj < state.get(j / 2)) return true;
        }
        if (j % 2 == 1 && j < n && j / 2 < state.size() && j - 1 < S.length) {
            if (Gj < S[j - 1] + state.get(j / 2)) return true;
        }
        if (j % 2 == 1 && j >= n && j / 2 < state.size() && j - n < A.length) {
            if (Gj < A[j - n] + state.get(j / 2)) return true;
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
        if (j % 2 == 0 && j / 2 < state.size()) {
            newValue = Math.max(newValue, state.get(j / 2));
        }
        if (j % 2 == 1 && j < n && j / 2 < state.size() && j - 1 < S.length) {
            newValue = Math.max(newValue, S[j - 1] + state.get(j / 2));
        }
        if (j % 2 == 1 && j >= n && j / 2 < state.size() && j - n < A.length) {
            newValue = Math.max(newValue, A[j - n] + state.get(j / 2));
        }

        state.set(j, newValue);
    }
}

class FastComponentsAlgorithm implements LatticeLinearPredicateFramework.LLPAlgorithm {
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

class BellmanFordAlgorithm implements LatticeLinearPredicateFramework.LLPAlgorithm {
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

class JohnsonAlgorithm implements LatticeLinearPredicateFramework.LLPAlgorithm {
    private final List<Integer>[] pre;
    private final int[][] w;

    public JohnsonAlgorithm(List<Integer>[] pre, int[][] w) {
        this.pre = pre;
        this.w = w;
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

        for (int i : pre[j]) {
            if (pj < state.get(i) - w[i][j]) {
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
        int maxPrice = Integer.MIN_VALUE;

        for (int i : pre[j]) {
            maxPrice = Math.max(maxPrice, state.get(i) - w[i][j]);
        }

        state.set(j, maxPrice);
    }
}

// ============= Test Case Generators =============

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

        public BenchmarkResult(String algorithmName, int problemSize, int numThreads, 
                             long executionTimeMs, boolean converged) {
            this.algorithmName = algorithmName;
            this.problemSize = problemSize;
            this.numThreads = numThreads;
            this.executionTimeMs = executionTimeMs;
            this.converged = converged;
        }

        @Override
        public String toString() {
            return String.format("%s | n=%d | threads=%d | time=%dms | converged=%b",
                algorithmName, problemSize, numThreads, executionTimeMs, converged);
        }
    }

    private final TestCaseGenerator generator;
    private final List<BenchmarkResult> results;

    public LLPBenchmark(long seed) {
        this.generator = new TestCaseGenerator(seed);
        this.results = new ArrayList<>();
    }

    public void runStableMarriageBenchmark(int n, int numThreads, long timeoutSeconds) {
        System.out.println("\n=== Benchmarking Stable Marriage (n=" + n + ", threads=" + numThreads + ") ===");
        
        TestCaseGenerator.StableMarriageInstance instance = generator.generateStableMarriage(n);
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
        
        BenchmarkResult result = new BenchmarkResult(
            "StableMarriage", n, numThreads, executionTime, converged);
        results.add(result);
        System.out.println(result);
    }

    public void runScanBenchmark(int n, int numThreads, long timeoutSeconds) {
        System.out.println("\n=== Benchmarking Scan (n=" + n + ", threads=" + numThreads + ") ===");
        
        int[] A = generator.generateScanInput(n);
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
        
        BenchmarkResult result = new BenchmarkResult(
            "Scan", n, numThreads, executionTime, converged);
        results.add(result);
        System.out.println(result);
    }

    public void runComponentsBenchmark(int n, double edgeProb, int numThreads, long timeoutSeconds) {
        System.out.println("\n=== Benchmarking Connected Components (n=" + n + ", threads=" + numThreads + ") ===");
        
        List<Integer>[] adj = generator.generateRandomGraph(n, edgeProb);
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
        
        BenchmarkResult result = new BenchmarkResult(
            "FastComponents", n, numThreads, executionTime, converged);
        results.add(result);
        System.out.println(result);
    }

    public void runBellmanFordBenchmark(int n, int avgDegree, int numThreads, long timeoutSeconds) {
        System.out.println("\n=== Benchmarking Bellman-Ford (n=" + n + ", threads=" + numThreads + ") ===");
        
        TestCaseGenerator.GraphInstance graph = generator.generateBellmanFordGraph(n, avgDegree);
        
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
        
        BenchmarkResult result = new BenchmarkResult(
            "BellmanFord", n, numThreads, executionTime, converged);
        results.add(result);
        System.out.println(result);
    }

    public void runJohnsonBenchmark(int n, int avgDegree, int numThreads, long timeoutSeconds) {
        System.out.println("\n=== Benchmarking Johnson (n=" + n + ", threads=" + numThreads + ") ===");
        
        TestCaseGenerator.GraphInstance graph = generator.generateJohnsonGraph(n, avgDegree);
        
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
        
        BenchmarkResult result = new BenchmarkResult(
            "Johnson", n, numThreads, executionTime, converged);
        results.add(result);
        System.out.println(result);
    }

    public void runComprehensiveBenchmark() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("COMPREHENSIVE LLP ALGORITHM BENCHMARK");
        System.out.println("=".repeat(80));

        int[] sizes = {10, 50, 100, 500};
        int[] threadCounts = {1, 2, 4, 8};
        long timeout = 30; // seconds

        // Stable Marriage Benchmarks
        System.out.println("\n" + "=".repeat(80));
        System.out.println("STABLE MARRIAGE BENCHMARKS");
        System.out.println("=".repeat(80));
        for (int size : sizes) {
            for (int threads : threadCounts) {
                if (threads <= size) {
                    runStableMarriageBenchmark(size, threads, timeout);
                }
            }
        }

        // Scan Benchmarks (use powers of 2)
        System.out.println("\n" + "=".repeat(80));
        System.out.println("SCAN BENCHMARKS");
        System.out.println("=".repeat(80));
        int[] scanSizes = {16, 64, 256, 1024};
        for (int size : scanSizes) {
            for (int threads : threadCounts) {
                runScanBenchmark(size, threads, timeout);
            }
        }

        // Connected Components Benchmarks
        System.out.println("\n" + "=".repeat(80));
        System.out.println("CONNECTED COMPONENTS BENCHMARKS");
        System.out.println("=".repeat(80));
        for (int size : sizes) {
            for (int threads : threadCounts) {
                if (threads <= size) {
                    runComponentsBenchmark(size, 0.1, threads, timeout);
                }
            }
        }

        // Bellman-Ford Benchmarks
        System.out.println("\n" + "=".repeat(80));
        System.out.println("BELLMAN-FORD BENCHMARKS");
        System.out.println("=".repeat(80));
        for (int size : sizes) {
            for (int threads : threadCounts) {
                if (threads <= size) {
                    runBellmanFordBenchmark(size, 4, threads, timeout);
                }
            }
        }

        // Johnson Benchmarks
        System.out.println("\n" + "=".repeat(80));
        System.out.println("JOHNSON BENCHMARKS");
        System.out.println("=".repeat(80));
        for (int size : sizes) {
            for (int threads : threadCounts) {
                if (threads <= size) {
                    runJohnsonBenchmark(size, 4, threads, timeout);
                }
            }
        }

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
        benchmark.runJohnsonBenchmark(100, 6, 4, 30);
        
        benchmark.printSummary();
        benchmark.exportToCSV("custom_benchmark_results.csv");
    }
}