function re_curve = recoverdata(now_curve)
load U.mat
    K = 20;
    re_curve = now_curve'*U(:,1:K)';
end