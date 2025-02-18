function PlotLineArrow(obj, x, y, lineColor)

% plot(x, y, 'o', 'Color', markerColor, 'MarkerFaceColor', markerColor);

posAxes = get(obj, 'Position');
posX = posAxes(1);
posY = posAxes(2);
width = posAxes(3);
height = posAxes(4);

limX = get(obj, 'Xlim');
limY = get(obj, 'Ylim');
minX = limX(1);
maxX = limX(2);
minY = limY(1);
maxY = limY(2);

xNew = posX + (x - minX) / (maxX - minX) * width;
yNew = posY + (y - minY) / (maxY - minY) * height;

annotation('arrow', xNew, yNew, 'color', lineColor);

