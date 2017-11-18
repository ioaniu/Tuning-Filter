function [cost,tuned] = compute_cost(sp_state_now)
    init_filter;
    
    for i = filter.start_point : filter.end_point
        if sp_state_now(i) > filter.threshold
            dist(i) = abs(sp_state_now(i) - filter.threshold);
        else
            dist(i) = 0;
        end
    end
    cost = norm(dist);
    
    if cost == 0
        tuned = true;
    else
        tuned = false;
    end
end