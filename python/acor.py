import numpy

def acor(value, bounds, nparams, nants=None, archive_size=None, maxit=1000,
         diverse=0.5, evap=0.85, seed=None):
    """
    Minimize the objective function using ACO-R.
    ACO-R stands for Ant Colony Optimization for Continuous Domains (Socha and
    Dorigo, 2008).
    Parameters:
    * value : function
        Returns the value of the objective function at a given parameter vector
    * bounds : list
        The bounds of the search space. If only two values are given, will
        interpret as the minimum and maximum, respectively, for all parameters.
        Alternatively, you can given a minimum and maximum for each parameter,
        e.g., for a problem with 3 parameters you could give
        `bounds = [min1, max1, min2, max2, min3, max3]`.
    * nparams : int
        The number of parameters that the objective function takes.
    * nants : int
        The number of ants to use in the search. Defaults to the number of
        parameters.
    * archive_size : int
        The number of solutions to keep in the solution archive. Defaults to
        10 x nants
    * maxit : int
        The number of iterations to run.
    * diverse : float
        Scalar from 0 to 1, non-inclusive, that controls how much better
        solutions are favored when constructing new ones.
    * evap : float
        The pheromone evaporation rate (evap > 0). Controls how spread out the
        search is.
    * seed : None or int
        Seed for the random number generator.
    Yields:
    * estimate : 1d-array
        The best estimate at each iteration
    """
    numpy.random.seed(seed)
    # Set the defaults for number of ants and archive size
    if nants is None:
        nants = nparams
    if archive_size is None:
        archive_size = 10 * nants
    # Check is giving bounds for each parameter or one for all
    bounds = numpy.array(bounds)
    if bounds.size == 2:
        low, high = bounds
        archive = numpy.random.uniform(low, high, (archive_size, nparams))
    else:
        archive = numpy.empty((archive_size, nparams))
        bounds = bounds.reshape((nparams, 2))
        for i, bound in enumerate(bounds):
            low, high = bound
            archive[:, i] = numpy.random.uniform(low, high, archive_size)
    # Compute the inital pheromone trail based on the objetive function value
    trail = numpy.fromiter((value(p) for p in archive), dtype=numpy.float)
    # Sort the archive
    order = numpy.argsort(trail)
    archive = [archive[i] for i in order]
    trail = trail[order].tolist()
    # The first of the archive is the best solution found
    yield archive[0]
    # Compute the weights (probabilities) of the solutions in the archive
    amp = 1. / (diverse * archive_size * numpy.sqrt(2 * numpy.pi))
    variance = 2 * diverse ** 2 * archive_size ** 2
    weights = amp * numpy.exp(-numpy.arange(archive_size) ** 2 / variance)
    weights /= numpy.sum(weights)
    for iteration in xrange(maxit):
        for k in xrange(nants):
            # Sample the propabilities to produce new estimates
            ant = numpy.empty(nparams, dtype=numpy.float)
            # 1. Choose a pdf from the archive
            pdf = numpy.searchsorted(
                numpy.cumsum(weights),
                numpy.random.uniform())
            for i in xrange(nparams):
                # 2. Get the mean and stddev of the chosen pdf
                mean = archive[pdf][i]
                std = (evap / (archive_size - 1)) * numpy.sum(
                    abs(p[i] - archive[pdf][i]) for p in archive)
                # 3. Sample the pdf until the samples are in bounds
                for atempt in xrange(100):
                    ant[i] = numpy.random.normal(mean, std)
                    if bounds.size == 2:
                        low, high = bounds
                    else:
                        low, high = bounds[i]
                    if ant[i] >= low and ant[i] <= high:
                        break
            pheromone = value(ant)
            # Place the new estimate in the archive
            place = numpy.searchsorted(trail, pheromone)
            if place == archive_size:
                continue
            trail.insert(place, pheromone)
            trail.pop()
            archive.insert(place, ant)
            archive.pop()
        yield archive[0]