# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
Cython optimized attribute mutation functions for NEAT.

This module provides high-performance implementations of attribute mutation operations,
avoiding Python's dynamic attribute lookups (getattr) and utilizing vectorized NumPy operations.

Key optimizations:
- Direct parameter passing instead of config getattr calls
- NumPy vectorized batch operations for processing multiple values
- C-level math functions for clamping
- nogil sections where possible for potential parallel execution
"""

import numpy as np
cimport numpy as np
from libc.math cimport fmax, fmin
from libc.stdlib cimport rand, RAND_MAX

# Initialize NumPy C API
np.import_array()


# =============================================================================
# Helper Functions
# =============================================================================

cdef inline double clamp(double value, double min_val, double max_val) noexcept nogil:
    """Clamp a value between min and max bounds."""
    return fmax(min_val, fmin(max_val, value))


cdef inline double c_random() noexcept nogil:
    """Generate a random double in [0, 1)."""
    return <double>rand() / (<double>RAND_MAX + 1.0)


# =============================================================================
# Float Attribute Mutations
# =============================================================================

cpdef double fast_float_mutate(
    double value,
    double mutate_rate,
    double replace_rate,
    double mutate_power,
    double min_value,
    double max_value,
    double init_mean,
    double init_stdev
):
    """
    Mutate a single float attribute value.

    This is a direct Cython implementation of FloatAttribute.mutate_value(),
    avoiding Python's getattr overhead by taking all parameters directly.

    Parameters:
        value: Current attribute value
        mutate_rate: Probability of mutation (gaussian perturbation)
        replace_rate: Probability of complete replacement with new random value
        mutate_power: Standard deviation for mutation gaussian
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        init_mean: Mean for initialization (replacement)
        init_stdev: Standard deviation for initialization

    Returns:
        Mutated value

    Logic:
        - If r < mutate_rate: perturb with gaussian noise
        - Elif r < mutate_rate + replace_rate: reinitialize with gaussian
        - Else: keep original value
    """
    cdef double r = np.random.random()
    cdef double new_value

    if r < mutate_rate:
        # Mutate: add gaussian noise
        new_value = value + np.random.normal(0.0, mutate_power)
        return clamp(new_value, min_value, max_value)

    if r < mutate_rate + replace_rate:
        # Replace: reinitialize with gaussian distribution
        new_value = np.random.normal(init_mean, init_stdev)
        return clamp(new_value, min_value, max_value)

    # No change
    return value


cpdef np.ndarray[np.float64_t, ndim=1] fast_float_mutate_batch(
    np.ndarray[np.float64_t, ndim=1] values,
    double mutate_rate,
    double replace_rate,
    double mutate_power,
    double min_value,
    double max_value,
    double init_mean,
    double init_stdev
):
    """
    Batch mutate multiple float attribute values using vectorized NumPy operations.

    This is the primary optimization function - it avoids Python for-loops by
    using NumPy's vectorized operations on entire arrays at once.

    Parameters:
        values: Array of current attribute values (will be copied, not modified in-place)
        mutate_rate: Probability of mutation for each value
        replace_rate: Probability of replacement for each value
        mutate_power: Standard deviation for mutation gaussian
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        init_mean: Mean for initialization (replacement)
        init_stdev: Standard deviation for initialization

    Returns:
        New array with mutated values

    Optimization strategy:
        1. Generate all random numbers at once
        2. Create boolean masks for each mutation type
        3. Apply mutations using masked array operations
        4. Single vectorized clamp at the end
    """
    cdef Py_ssize_t n = values.shape[0]

    if n == 0:
        return values.copy()

    # Create output array (don't modify input)
    cdef np.ndarray[np.float64_t, ndim=1] result = values.copy()

    # Generate all random values at once
    cdef np.ndarray[np.float64_t, ndim=1] rand_vals = np.random.random(n)

    # Create masks for different operations
    cdef np.ndarray[np.npy_bool, ndim=1, cast=True] mutate_mask = rand_vals < mutate_rate
    cdef np.ndarray[np.npy_bool, ndim=1, cast=True] replace_mask = (
        (rand_vals >= mutate_rate) & (rand_vals < mutate_rate + replace_rate)
    )

    # Count how many values need gaussian noise
    cdef Py_ssize_t n_mutate = np.sum(mutate_mask)
    cdef Py_ssize_t n_replace = np.sum(replace_mask)

    # Apply mutations
    if n_mutate > 0:
        # Generate gaussian noise only for values that will be mutated
        result[mutate_mask] += np.random.normal(0.0, mutate_power, n_mutate)

    if n_replace > 0:
        # Generate new random values for replacements
        result[replace_mask] = np.random.normal(init_mean, init_stdev, n_replace)

    # Clamp all values at once
    np.clip(result, min_value, max_value, out=result)

    return result


cpdef np.ndarray[np.float64_t, ndim=1] fast_float_init_batch(
    Py_ssize_t n,
    double init_mean,
    double init_stdev,
    double min_value,
    double max_value,
    str init_type = 'gaussian'
):
    """
    Batch initialize multiple float attribute values.

    Parameters:
        n: Number of values to initialize
        init_mean: Mean for gaussian initialization
        init_stdev: Standard deviation for gaussian initialization
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        init_type: 'gaussian' or 'uniform'

    Returns:
        Array of initialized values
    """
    cdef np.ndarray[np.float64_t, ndim=1] result
    cdef double uniform_min, uniform_max

    if n <= 0:
        return np.empty(0, dtype=np.float64)

    if init_type == 'gaussian' or init_type == 'normal':
        result = np.random.normal(init_mean, init_stdev, n)
    elif init_type == 'uniform':
        # For uniform, use mean +/- 2*stdev as range, clamped to min/max
        uniform_min = fmax(min_value, init_mean - 2.0 * init_stdev)
        uniform_max = fmin(max_value, init_mean + 2.0 * init_stdev)
        result = np.random.uniform(uniform_min, uniform_max, n)
    else:
        # Default to gaussian
        result = np.random.normal(init_mean, init_stdev, n)

    # Clamp values
    np.clip(result, min_value, max_value, out=result)

    return result


# =============================================================================
# Integer Attribute Mutations
# =============================================================================

cpdef long fast_int_mutate(
    long value,
    double mutate_rate,
    double replace_rate,
    double mutate_power,
    long min_value,
    long max_value
):
    """
    Mutate a single integer attribute value.

    Parameters:
        value: Current attribute value
        mutate_rate: Probability of mutation
        replace_rate: Probability of replacement
        mutate_power: Standard deviation for mutation gaussian
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Returns:
        Mutated integer value
    """
    cdef double r = np.random.random()
    cdef long new_value

    if r < mutate_rate:
        # Mutate: add rounded gaussian noise
        new_value = value + <long>round(np.random.normal(0.0, mutate_power))
        return max(min_value, min(max_value, new_value))

    if r < mutate_rate + replace_rate:
        # Replace: random integer in range
        return np.random.randint(min_value, max_value + 1)

    return value


cpdef np.ndarray[np.int64_t, ndim=1] fast_int_mutate_batch(
    np.ndarray[np.int64_t, ndim=1] values,
    double mutate_rate,
    double replace_rate,
    double mutate_power,
    long min_value,
    long max_value
):
    """
    Batch mutate multiple integer attribute values.

    Parameters:
        values: Array of current integer values
        mutate_rate: Probability of mutation for each value
        replace_rate: Probability of replacement for each value
        mutate_power: Standard deviation for mutation gaussian
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Returns:
        New array with mutated integer values
    """
    cdef Py_ssize_t n = values.shape[0]

    if n == 0:
        return values.copy()

    # Create output array
    cdef np.ndarray[np.int64_t, ndim=1] result = values.copy()

    # Generate all random values at once
    cdef np.ndarray[np.float64_t, ndim=1] rand_vals = np.random.random(n)

    # Create masks
    cdef np.ndarray[np.npy_bool, ndim=1, cast=True] mutate_mask = rand_vals < mutate_rate
    cdef np.ndarray[np.npy_bool, ndim=1, cast=True] replace_mask = (
        (rand_vals >= mutate_rate) & (rand_vals < mutate_rate + replace_rate)
    )

    cdef Py_ssize_t n_mutate = np.sum(mutate_mask)
    cdef Py_ssize_t n_replace = np.sum(replace_mask)

    # Apply mutations
    if n_mutate > 0:
        # Add rounded gaussian noise
        result[mutate_mask] += np.round(
            np.random.normal(0.0, mutate_power, n_mutate)
        ).astype(np.int64)

    if n_replace > 0:
        # Random integers in range
        result[replace_mask] = np.random.randint(min_value, max_value + 1, n_replace)

    # Clamp
    np.clip(result, min_value, max_value, out=result)

    return result


# =============================================================================
# Bool Attribute Mutations
# =============================================================================

cpdef bint fast_bool_mutate(
    bint value,
    double mutate_rate,
    double rate_to_true_add,
    double rate_to_false_add
):
    """
    Mutate a single boolean attribute value.

    The effective mutation rate depends on the current value:
    - If value is True: effective_rate = mutate_rate + rate_to_false_add
    - If value is False: effective_rate = mutate_rate + rate_to_true_add

    When mutation occurs, the new value is random (50/50 chance).

    Parameters:
        value: Current boolean value
        mutate_rate: Base mutation probability
        rate_to_true_add: Additional probability when value is False
        rate_to_false_add: Additional probability when value is True

    Returns:
        Mutated boolean value
    """
    cdef double effective_rate = mutate_rate

    if value:
        effective_rate += rate_to_false_add
    else:
        effective_rate += rate_to_true_add

    if effective_rate > 0.0:
        if np.random.random() < effective_rate:
            # Mutate to random value (not guaranteed to change)
            return np.random.random() < 0.5

    return value


cpdef np.ndarray[np.npy_bool, ndim=1] fast_bool_mutate_batch(
    np.ndarray[np.npy_bool, ndim=1] values,
    double mutate_rate,
    double rate_to_true_add,
    double rate_to_false_add
):
    """
    Batch mutate multiple boolean attribute values.

    Parameters:
        values: Array of current boolean values
        mutate_rate: Base mutation probability
        rate_to_true_add: Additional probability when value is False
        rate_to_false_add: Additional probability when value is True

    Returns:
        New array with mutated boolean values
    """
    cdef Py_ssize_t n = values.shape[0]

    if n == 0:
        return values.copy()

    # Create output array
    cdef np.ndarray[np.npy_bool, ndim=1] result = values.copy()

    # Generate random values
    cdef np.ndarray[np.float64_t, ndim=1] rand_vals = np.random.random(n)

    # Calculate effective mutation rates based on current values
    # For True values: effective_rate = mutate_rate + rate_to_false_add
    # For False values: effective_rate = mutate_rate + rate_to_true_add
    cdef np.ndarray[np.float64_t, ndim=1] effective_rates = np.where(
        values,
        mutate_rate + rate_to_false_add,
        mutate_rate + rate_to_true_add
    )

    # Find which values should mutate
    cdef np.ndarray[np.npy_bool, ndim=1, cast=True] mutate_mask = rand_vals < effective_rates
    cdef Py_ssize_t n_mutate = np.sum(mutate_mask)

    if n_mutate > 0:
        # Generate new random boolean values for mutations
        result[mutate_mask] = np.random.random(n_mutate) < 0.5

    return result


cpdef np.ndarray[np.npy_bool, ndim=1] fast_bool_init_batch(
    Py_ssize_t n,
    str default = 'random'
):
    """
    Batch initialize multiple boolean attribute values.

    Parameters:
        n: Number of values to initialize
        default: 'true', 'false', or 'random'

    Returns:
        Array of initialized boolean values
    """
    cdef str default_lower = default.lower()

    if n <= 0:
        return np.empty(0, dtype=np.bool_)

    if default_lower in ('1', 'on', 'yes', 'true'):
        return np.ones(n, dtype=np.bool_)
    elif default_lower in ('0', 'off', 'no', 'false'):
        return np.zeros(n, dtype=np.bool_)
    else:  # 'random', 'none', or anything else
        return np.random.random(n) < 0.5


# =============================================================================
# Combined Batch Operations for Genome Mutation
# =============================================================================

cpdef tuple fast_mutate_genome_floats(
    np.ndarray[np.float64_t, ndim=1] weights,
    np.ndarray[np.float64_t, ndim=1] biases,
    np.ndarray[np.float64_t, ndim=1] responses,
    double weight_mutate_rate,
    double weight_replace_rate,
    double weight_mutate_power,
    double weight_min,
    double weight_max,
    double weight_init_mean,
    double weight_init_stdev,
    double bias_mutate_rate,
    double bias_replace_rate,
    double bias_mutate_power,
    double bias_min,
    double bias_max,
    double bias_init_mean,
    double bias_init_stdev,
    double response_mutate_rate,
    double response_replace_rate,
    double response_mutate_power,
    double response_min,
    double response_max,
    double response_init_mean,
    double response_init_stdev
):
    """
    Batch mutate all float attributes of a genome in one call.

    This function mutates weights, biases, and responses together,
    which is more efficient than calling separate batch functions.

    Returns:
        Tuple of (mutated_weights, mutated_biases, mutated_responses)
    """
    cdef np.ndarray[np.float64_t, ndim=1] new_weights
    cdef np.ndarray[np.float64_t, ndim=1] new_biases
    cdef np.ndarray[np.float64_t, ndim=1] new_responses

    new_weights = fast_float_mutate_batch(
        weights, weight_mutate_rate, weight_replace_rate,
        weight_mutate_power, weight_min, weight_max,
        weight_init_mean, weight_init_stdev
    )

    new_biases = fast_float_mutate_batch(
        biases, bias_mutate_rate, bias_replace_rate,
        bias_mutate_power, bias_min, bias_max,
        bias_init_mean, bias_init_stdev
    )

    new_responses = fast_float_mutate_batch(
        responses, response_mutate_rate, response_replace_rate,
        response_mutate_power, response_min, response_max,
        response_init_mean, response_init_stdev
    )

    return (new_weights, new_biases, new_responses)
