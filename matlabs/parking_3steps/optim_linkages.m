function linkages = optim_linkages()
    
    ipair = 1;
    linkages(ipair).left.phase = 1;
    linkages(ipair).right.phase = 2;
    linkages(ipair).min = [0.0; 0.0; 0.0; 0.0; 0.0];
    linkages(ipair).max = [0.0; 0.0; 0.0; 0.0; 0.0];
    
    ipair = 2;
    linkages(ipair).left.phase = 2;
    linkages(ipair).right.phase = 3;
    linkages(ipair).min = [0.0; 0.0; 0.0; 0.0; 0.0];
    linkages(ipair).max = [0.0; 0.0; 0.0; 0.0; 0.0];
    
end