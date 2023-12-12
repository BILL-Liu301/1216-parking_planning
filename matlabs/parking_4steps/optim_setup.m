function setup = optim_setup(limits, guess, linkages)
    setup.name = 'optim_parking';
    setup.funcs.dae = 'optim_dae';
    setup.funcs.cost = 'optim_cost';
    setup.funcs.link = 'optim_link';
    setup.derivatives = 'finite-difference';
    setup.checkDerivatives = 0;
    setup.limits = limits;
    setup.guess = guess;
    setup.linkages = linkages;
    setup.autoscale = 'off';
    setup.mesh.tolerance = 1e-6;
    setup.mesh.iteration = 5;
    setup.mesh.nodesPerInterval.min = 4;
    setup.mesh.nodesPerInterval.max = 12;
end