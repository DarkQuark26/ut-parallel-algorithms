package hw.hw4;

import java.util.concurrent.atomic.AtomicIntegerArray;

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

    public int get(int index) {
        return state.get(index);
    }

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

    boolean forbidden(ThreadContext ctx);

    int advance(ThreadContext ctx);

    default boolean hasConverged(GlobalState G) {
        return false;
    }
    
}

class LLPRunner {
    private final int n;
    private final GlobalState G;
    private final LLPAlgorithm algorithm;
    private final 

}

}
