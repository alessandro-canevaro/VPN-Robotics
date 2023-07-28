%clear all
close all
load('fin_values2.mat')


fin = round(fin);
%surf(fin);
surf(fin, 'EdgeColor', 'flat');
grayColor = [.7 .7 .7];
set(gca,'Color',grayColor)
xlim([2 34])
ylim([2 34])

colorbar
hold on
h = scatter3(6,30,-81,'filled','MarkerFaceColor','red');
h.SizeData = 150;
view(0, 90)