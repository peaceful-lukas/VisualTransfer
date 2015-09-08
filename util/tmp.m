
% c1 = 2;
% c2 = 9;


U_c1 = U(:, protoStartIdx(c1)+1:protoStartIdx(c1+1));
U_c2 = U(:, protoStartIdx(c2)+1:protoStartIdx(c2+1));

figure;
[coeff,red_U_c1, eigval] = princomp(U_c1');
red_U_c1 = red_U_c1(:, 1:2);
gplot(param.knnGraphs{c1}, red_U_c1)

figure;
[coeff,red_U_c2, eigval] = princomp(U_c2');
red_U_c2 = red_U_c2(:, 1:2);
gplot(param.knnGraphs{c2}, red_U_c2)


[coeff,red_U_c1c2, eigval] = princomp([U_c1'; U_c2']);
red_U_c1c2 = red_U_c1c2(:, 1:3);
A_c1c2 = zeros(21, 21);
A_c1c2(1:12, 1:12) = param.knnGraphs{c1};
A_c1c2(13:21, 13:21) = param.knnGraphs{c2};
gplot(A_c1c2, red_U_c1c2);




org_U_c1 = classProtos(:, protoStartIdx(c1)+1:protoStartIdx(c1+1));
org_U_c2 = classProtos(:, protoStartIdx(c2)+1:protoStartIdx(c2+1));


[coeff,red_U_c1, eigval] = princomp(org_U_c1');
red_U_c1 = red_U_c1(:, 1:2);
gplot(param.knnGraphs{c1}, red_U_c1)


[coeff,red_U_c2, eigval] = princomp(org_U_c2');
red_U_c2 = red_U_c2(:, 1:2);
gplot(param.knnGraphs{c2}, red_U_c2)


[coeff,red_U_c1c2, eigval] = princomp([org_U_c1'; org_U_c2']);
red_U_c1c2 = red_U_c1c2(:, 1:2);
A_c1c2 = zeros(21, 21);
A_c1c2(1:12, 1:12) = param.knnGraphs{c1};
A_c1c2(13:21, 13:21) = param.knnGraphs{c2};
gplot(A_c1c2, red_U_c1c2);




% img = imread('peppers.png');             %# Load a sample image
% scatter(rand(1,20)-0.5,rand(1,20)-0.5);  %# Plot some random data
hold on;                                 %# Add to the plot
image([-0.1 0.1],[0.1 -0.1],img);        %# Plot the image



for n=1:9
    u_proto = U_c2(:, n);
    [~, idx] = sort(DS.D'*W'*u_proto, 'descend');
    imshow(trI{idx(4)});
    disp(n);
    pause
end












figure;
[coeff,red_U_c2, eigval] = princomp(U_c2');
red_U_c2 = red_U_c2(:, 1:2);
gplot(param.knnGraphs{c2}, red_U_c2)

AXES = gca;

for i=1:9
    % if 10*rand(1)>4
    axes(AXES);

    x_bounds = xlim;
    x_bounds(1) = x_bounds(1)*5/4;
    x_bounds(2) = x_bounds(2)*5/4;
    x_range = x_bounds(2) - x_bounds(1);
    n_X = (red_U_c2(i, 1) - x_bounds(1)) / x_range;

    y_bounds = ylim;
    y_bounds(1) = y_bounds(1)*5/4;
    y_bounds(2) = y_bounds(2)*5/4;
    y_range = y_bounds(2) - y_bounds(1);
    n_Y = (red_U_c2(i, 2) - y_bounds(1)) / y_range;

    img = trI{i};
    [height width] = size(img);


    axes('position', [n_X-0.05, n_Y-0.05, 0.1, 0.1]);
    imshow(img);
% end
end



