def _slow_negative_log_likelihood(args, Y, D, X, Z):
    """ Negative Log-likelihood function of the generalized Roy model.
    """
    # Distribute arguments
    Y1_coeffs, Y0_coeffs, C_coeffs, choice_coeffs, U1_var, U0_var, U1V_rho, \
    U0V_rho, V_var, U1_sd, U0_sd, V_sd = _distribute_arguments(args)

    # Auxiliary objects
    num_agents = Y.shape[0]

    # Initialize containers
    likl = np.tile(np.nan, num_agents)
    choice_idx = np.tile(np.nan, num_agents)

    # Likelihood construction.
    for i in range(num_agents):

        g = np.concatenate((X[i, :], Z[i, :]))
        choice_idx[i] = np.dot(choice_coeffs, g)

        # Select outcome information
        if D[i] == 1.00:

            coeffs, rho, sd = Y1_coeffs, U1V_rho, U1_sd
        else:
            coeffs, rho, sd = Y0_coeffs, U0V_rho, U0_sd

        arg_one = (Y[i] - np.dot(coeffs, X[i, :])) / sd
        arg_two = (choice_idx[i] - rho * V_sd * arg_one) / \
                  np.sqrt((1.0 - rho ** 2) * V_var)

        pdf_evals, cdf_evals = norm.pdf(arg_one), norm.cdf(arg_two)

        if D[i] == 1.0:
            contrib = (1.0 / float(sd)) * pdf_evals * cdf_evals
        else:
            contrib = (1.0 / float(sd)) * pdf_evals * (1.0 - cdf_evals)

        likl[i] = contrib

    # Transformations.
    likl = -np.mean(np.log(np.clip(likl, 1e-20, np.inf)))

    # Quality checks.
    assert (isinstance(likl, float))
    assert (np.isfinite(likl))

    # Finishing.
    return likl

def _fast_negative_log_likelihood(args, Y, D, X, Z):
    """ Negative Log-likelihood function of the Generalized Roy Model.
    """
    # Distribute arguments
    Y1_coeffs, Y0_coeffs, C_coeffs, choice_coeffs, U1_var, U0_var, U1V_rho, \
    U0V_rho, V_var, U1_sd, U0_sd, V_sd = _distribute_arguments(args)

    # Likelihood construction.
    G = np.concatenate((X, Z), axis=1)
    choice_idx = np.dot(choice_coeffs, G.T)

    arg_one = D * (Y - np.dot(Y1_coeffs, X.T)) / U1_sd + \
             (1 - D) * (Y - np.dot(Y0_coeffs, X.T)) / U0_sd

    arg_two = D * (choice_idx - V_sd * U1V_rho * arg_one) / np.sqrt(
        (1.0 - U1V_rho ** 2) * V_var) + \
        (1 - D) * (choice_idx - V_sd * U0V_rho * arg_one) / np.sqrt(
                 (1.0 - U0V_rho ** 2) * V_var)

    pdf_evals, cdf_evals = norm.pdf(arg_one), norm.cdf(arg_two)

    likl = D * (1.0 / U1_sd) * pdf_evals * cdf_evals + \
           (1 - D) * (1.0 / U0_sd) * pdf_evals * (1.0 - cdf_evals)

    # Transformations.
    likl = -np.mean(np.log(np.clip(likl, 1e-20, np.inf)))

    # Quality checks.
    assert (isinstance(likl, float))
    assert (np.isfinite(likl))

    # Finishing.
    return likl