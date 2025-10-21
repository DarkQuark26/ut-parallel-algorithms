package hw.hw4;

public class LatticeLinearPredicate {
// Core interfaces for Lattice Linear Predicate algorithms

/**
 * Represents a lattice element (typically a global state or cut)
 * @param <T> the type of the lattice element
 */
public interface LatticeElement<T> {
    /**
     * Compares this element with another in the lattice partial order
     * @param other the element to compare with
     * @return true if this element is less than or equal to other
     */
    boolean isLessThanOrEqual(T other);
    
    /**
     * Computes the join (least upper bound) with another element
     * @param other the element to join with
     * @return the join of this and other
     */
    T join(T other);
    
    /**
     * Computes the meet (greatest lower bound) with another element
     * @param other the element to meet with
     * @return the meet of this and other
     */
    T meet(T other);
    
    /**
     * Creates a deep copy of this lattice element
     * @return a copy of this element
     */
    T copy();
}

/**
 * Represents a predicate that can be evaluated on lattice elements
 * @param <T> the type of lattice element
 */
public interface Predicate<T extends LatticeElement<T>> {
    /**
     * Evaluates the predicate on a given lattice element
     * @param element the element to evaluate
     * @return true if the predicate holds on the element
     */
    boolean evaluate(T element);
    
    /**
     * Checks if this is a linear predicate
     * @return true if the predicate is linear
     */
    boolean isLinear();
}

/**
 * Represents a lattice structure for predicate detection
 * @param <T> the type of lattice elements
 */
public interface Lattice<T extends LatticeElement<T>> {
    /**
     * Gets the bottom element of the lattice
     * @return the minimal element
     */
    T getBottom();
    
    /**
     * Gets the top element of the lattice
     * @return the maximal element
     */
    T getTop();
    
    /**
     * Returns all immediate successors of a given element
     * @param element the element whose successors to find
     * @return an iterable of successor elements
     */
    Iterable<T> getSuccessors(T element);
    
    /**
     * Returns all immediate predecessors of a given element
     * @param element the element whose predecessors to find
     * @return an iterable of predecessor elements
     */
    Iterable<T> getPredecessors(T element);
    
    /**
     * Checks if the lattice is finite
     * @return true if the lattice has finitely many elements
     */
    boolean isFinite();
}

/**
 * Main interface for lattice linear predicate detection algorithms
 * @param <T> the type of lattice elements
 */
public interface LatticePredicate Algorithm<T extends LatticeElement<T>> {
    /**
     * Detects if the predicate holds on any element in the lattice
     * @param lattice the lattice to search
     * @param predicate the predicate to detect
     * @return true if the predicate holds on at least one element
     */
    boolean detect(Lattice<T> lattice, Predicate<T> predicate);
    
    /**
     * Finds all lattice elements where the predicate holds
     * @param lattice the lattice to search
     * @param predicate the predicate to detect
     * @return an iterable of elements satisfying the predicate
     */
    Iterable<T> findAll(Lattice<T> lattice, Predicate<T> predicate);
    
    /**
     * Finds the first (minimal) element where the predicate holds
     * @param lattice the lattice to search
     * @param predicate the predicate to detect
     * @return the minimal element satisfying the predicate, or null if none exists
     */
    T findFirst(Lattice<T> lattice, Predicate<T> predicate);
    
    /**
     * Gets the name of this algorithm
     * @return the algorithm name
     */
    String getAlgorithmName();
}

/**
 * Factory for creating lattice predicate algorithm instances
 */
public interface LatticePredicateAlgorithmFactory {
    /**
     * Creates an algorithm instance for the given lattice type
     * @param <T> the type of lattice elements
     * @return a new algorithm instance
     */
    <T extends LatticeElement<T>> LatticePredicateAlgorithm<T> createAlgorithm();
    
    /**
     * Gets the supported algorithm types
     * @return description of supported algorithms
     */
    String getSupportedTypes();
}
}
