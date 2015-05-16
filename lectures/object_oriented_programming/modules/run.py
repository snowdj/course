def _choose(self, y, p1, p2):
        """ Choose utility-maximizing bundle.
        """
        # Antibugging
        integrity_checks('_choose_rational_in', y, p1, p2)

        # Determine starting values
        x0 = np.array([(0.5 * y) / p1, (0.5 * y) / p2])

        # Construct budget constraint
        constraint_divergence = dict()

        constraint_divergence['type'] = 'eq'

        constraint_divergence['args'] = (p1, p2)

        constraint_divergence['fun'] = self._constraint

        constraints = [constraint_divergence, ]

        # Call constraint-optimizer. Of course, we could determine the
        # optimal bundle directly, but I wanted to illustrate the use of
        # a constraint optimization algorithm to you.
        rslt = minimize(self._criterion, x0, method='SLSQP',
                        constraints=constraints)

        # Check for convergence
        assert (rslt['success'] == True)

        # Transformation of result.
        x = rslt['x'] ** 2

        # Type conversion
        x = x.tolist()

        # Quality Checks
        integrity_checks('_choose_rational_out', x)

        # Finishing
        return x
