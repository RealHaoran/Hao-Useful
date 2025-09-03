% CLT 验证 - 6种分布
clear; clc;

num_experiments = 10000;
sample_sizes = [1, 2, 5, 10, 30, 50, 100];

% 分布列表
distributions = {
    'Uniform(0,1)', @(n) rand(n, 1);
    'Exponential(λ=1)', @(n) exprnd(1, n, 1);
    'Binomial(n=1,p=0.5)', @(n) binornd(1, 0.5, n, 1);
    'Poisson(λ=3)', @(n) poissrnd(3, n, 1);
    'Chi-squared(df=2)', @(n) chi2rnd(2, n, 1);
    'Beta(a=2,b=5)', @(n) betarnd(2, 5, n, 1);
};

figure('Position', [100, 100, 1800, 900]);
tiledlayout(length(distributions), length(sample_sizes), 'Padding', 'compact');

for d = 1:size(distributions,1)
    dist_name = distributions{d, 1};
    dist_func = distributions{d, 2};
    for s = 1:length(sample_sizes)
        n = sample_sizes(s);
        sample_means = zeros(num_experiments, 1);
        for i = 1:num_experiments
            sample_data = dist_func(n);
            sample_means(i) = mean(sample_data);
        end

        % 子图绘制
        nexttile;
        histogram(sample_means, 30, 'Normalization', 'pdf', 'FaceColor', [0.5 0.8 1]); hold on;

        % 理论正态分布拟合
        mu = mean(sample_means);
        sigma = std(sample_means);
        x_vals = linspace(min(sample_means), max(sample_means), 100);
        y_vals = normpdf(x_vals, mu, sigma);
        plot(x_vals, y_vals, 'r--', 'LineWidth', 1.5);

        % 标题与坐标轴
        title(sprintf('%s\nn = %d', dist_name, n), 'FontSize', 10);
        xlabel('Sample Mean');
        ylabel('Density');
    end
end

sgtitle('CLT Verification for 6 Distributions', 'FontSize', 16);