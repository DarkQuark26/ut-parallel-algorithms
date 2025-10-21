package hw.hw4;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.function.*;

public class LatticeLinearPredicateAlgorithm {

class GlobalState {
    // Maybe generalize to be interface? With this global state implementing those
    // that can use integer arrays to represent global state

    private final AtomicIntegerArray state;
    private final int n;

    public GlobalState(int n) {
        this.n = n;
        this.state = new AtomicIntegerArray(n);
    }

    /**
     * Return G[j]
     * 
     * @param index j
     * @return G[j]
     */
    public int get(int index) {
        return state.get(index);
    }

    /**
     * 
     * @param index
     * @param value
     */
    public void set(int index, int value) {
        state.set(index, value);
    }

    public boolean compareAndSet(int index, int expect, int update) {
        return state.compareAndSet(index, expect, update);
    }

    public int size() {
        return n;
    }
    /**
     * 
     * @return snapshot, a view of G as a int[] at the time this was run
     * May not be current due to other threads updating the array while
     * traversing, so be careful reading too much into it
     */
    public int[] snapshot() {
        int[] snapshot = new int[n];
        for (int i = 0; i < n; i++) {
            snapshot[i] = state.get(i);
        }
        return snapshot;
    }
}

/**
 * Class the controls specific element in G
 */
class ThreadContext {
    public final int j;
    public final GlobalState G;

    public ThreadContext(int j, GlobalState G) {
        this.j = j;
        this.G = G;
    }
}

public interface LLPAlgorithm {
    // Add ensure for alternative to forbidden and advance?
    /**
     * Initializers a thread in the default state
     * 
     * @param ctx
     */
    void init(ThreadContext ctx);

    /**
     * Check if state G[j] is forbidden
     * 
     * @param ctx ThreadContext object that tracks index in G
     * @return True if state is forbidden
     */
    boolean forbidden(ThreadContext ctx);

    /**
     * Advance state G[j]
     * 
     * @param ctx
     * @return New state for G[j]
     */
    int advance(ThreadContext ctx);

    // Could use an auxiliary array to track if all threads are
    // not advancing, or just wait and check if G hasn't been
    // modified in a while?
    default boolean hasConverged(GlobalState G) {
        return false; 
    }
    
}

class LLPRunner {
    private final int n; // Number of elements in global state
    private final int numPlatformThreads; // Number of platform threads to use
    private final GlobalState G; // Global state
    private final LLPAlgorithm algorithm; // Algorithm for particular problem
    private final ExecutorService executor; // Manages threads
    private final AtomicBoolean running; // Flag for all platform threads to check state of algorithm
    private final AtomicInteger activeThreads; // Counter for number of active threads
    private final CountDownLatch initLatch; // Used 
    //private final AtomicIntegerArray forbiddenFlags; // Use to track which elements are currrently forbidden, termination behavior
    
    
    public LLPRunner(int n, LLPAlgorithm algorithm) {
        this(n, algorithm, Runtime.getRuntime().availableProcessors());
    }

    public LLPRunner(int n, LLPAlgorithm algorithm, int numPlatformThreads) {
        this.n = n;
        this.numPlatformThreads = Math.min(numPlatformThreads, n);
        this.G = new GlobalState(n);
        this.algorithm = algorithm;
        this.executor = Executors.newFixedThreadPool(this.numPlatformThreads);
        this.running = new AtomicBoolean(false);
        this.activeThreads = new AtomicInteger(0);
        this.initLatch = new CountDownLatch(n);
    }

    public void start() {
        // Initalize logical threads (to be used by platofrm threads)
        for (int j = 0; j < n; j++) { // Not parallel since assume runtime
            final int threadID = j;
            ThreadContext ctx = new ThreadContext(threadID, G);
            algorithm.init(ctx);
            initLatch.countDown(); // Add in case want to parallelize further later
        }

        // Start platform threads
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
            initLatch.await(); // Waits for thread initialization to complete
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            activeThreads.decrementAndGet();
            return;
        }
        
        // Uses strided execution, i.e thread j runs G[j], G[j+numPlatformThreads], G[j+2*numPlatformThreads], etc.
        while (running.get()) {
            for (int j = platformThreadId; j < n; j += numPlatformThreads) {
                ThreadContext ctx = new ThreadContext(j ,G);

                if (algorithm.forbidden(ctx)) {
                    int newValue = algorithm.advance(ctx);
                    G.set(j ,newValue);
                }
            }
            Thread.yield(); // Just in case number of platform threads is larger than number of physical
        }
        activeThreads.decrementAndGet();
    }

}

}

/*
 * Stable Marriage Problem
 * Parallel Prefix problem
 * Finding connected components of an undirected graph (Fast Algorithm)
 * Bellman-Ford Algorithm
 * Johnson’s algorithm for shortest path
 * Boruvka’s Algorithm
 */