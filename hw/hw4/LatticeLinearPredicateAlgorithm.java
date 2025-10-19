package hw.hw4;

import java.util.concurrent.atomic.AtomicIntegerArray;

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

    public int[] snapshot() {
        int[] snapshot = new int[n];
        for (int i = 0; i < n; i++) {
            snapshot[i] = state.get(i);
        }
        return snapshot;
    }
}

class ThreadContext {
    public final int j;
    public final GlobalState G;

    public ThreadContext(int j, GlobalState G) {
        this.j = j;
        this.G = G;
    }
}

public interface LLPAlgorithm {
    // Add ensure for alternative to orbidden and advance?
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
    private final GlobalState G; //
    private final LLPAlgorithm algorithm;
    
    private final int numPlatformThreads; // Number of platform threads to use
    
    
    public LLPRunner(int n, LLPAlgorithm algorithm) {
        this(n, algorithm, Runtime.getRuntime().availableProcessors());
    }


    public void start() {
        // Initalize logical threads (to be used by platofrm threads)
        for (int j = 0; j < n; j++) {
            final int threadID = j;
            ThreadContext ctx = new ThreadContext(threadID, G);
            algorithm.init(ctx);
            initLatch.countDown();
        }

        // Populate work queue
        for (int j = 0; j < n; j++) {
            workQueue.offer(j)
        }

        // Start platform threads
        for (int p = 0; p < numPlatformThreads; p++) {
            runner.submit(this::platformThreadLoop)
        }
    }

}

}
