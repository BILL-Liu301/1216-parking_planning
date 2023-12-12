function sizes = util_sizes(output)
    
    theta = output.solution(4).state(:, 3);
    ind = find(abs(theta - pi/2) > 0.1);

    sizes = [size(output.solution(1).state, 1)/2, size(output.solution(2).state, 1)/2, size(output.solution(3).state, 1)/2, ind(end)];
    sizes = round(sizes);
end