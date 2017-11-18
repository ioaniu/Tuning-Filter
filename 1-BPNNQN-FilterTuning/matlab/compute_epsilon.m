function res = compute_epsilon(count_now,count_total)    
    res = max(0.99 - count_now / count_total,0.1);
end