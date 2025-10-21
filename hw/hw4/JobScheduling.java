package hw.hw4;

import java.util.List;

import hw.hw4.LatticeLinearPredicateAlgorithm.GlobalState;
import hw.hw4.LatticeLinearPredicateAlgorithm.LLPAlgorithm;
import hw.hw4.LatticeLinearPredicateAlgorithm.ThreadContext;

public class JobScheduling {

class JobSchedulingLLP implements LLPAlgorithm {
    private final int[] t; // execution times
    private final List<List<Integer>> pre; // precedence constraints

    public JobSchedulingLLP(int[] t, List<List<Integer>> pre) {
        this.t = t;
        this.pre = pre;
    }

    @Override
    public void init(ThreadContext ctx) {
        ctx.G.set(ctx.j, t[ctx.j]);
    }

    @Override
    public boolean forbidden(ThreadContext ctx) {
        int currentValue = ctx.G.get(ctx.j);
        int maxPredecessor = computeMax(ctx);
        return currentValue < maxPredecessor;
    }

    @Override
    public int advance(ThreadContext ctx) {
        return computeMax(ctx);
    }

    private int computeMax(ThreadContext ctx) {
        int max = t[ctx.j];
        for (int i : pre.get(ctx.j)) {
            max = Math.max(max, ctx.G.get(i) + t[ctx.j]);
        }
        return max;
    }

    @Override
    public boolean hasConverged(GlobalState G) {
        // Check if all threads are stable (no forbidden states)
        for (int j = 0; j < G.size(); j++) {
            ThreadContext ctx = new ThreadContext(j, G);
            if (forbidden(ctx)) {
                return false;
            }
        }
        return true;
    }
}

}
