function link = optim_link(sol)
    xf_left = sol.left.state;
    x0_right = sol.right.state;
    link = xf_left - x0_right;
end