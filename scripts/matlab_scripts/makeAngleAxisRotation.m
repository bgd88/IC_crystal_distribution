% function R = makeAngleAxisRotation(angle, u);
%
% Make the 3x3 rotation matrix for rotating angle degrees about axis u.
%
% angle: rotation angle in degrees
% u: 3x1 vector representing the axis of rotation

function R = makeAngleAxisRotation(angle, u)

angle = pi/180*angle;
u = reshape(u, 3, 1);
u = u ./ norm(u);

c = cos(angle);
s = sin(angle);
R = c*eye(3) + (1-c)*(u*u') + s* [0 -u(3) u(2); u(3) 0 -u(1); -u(2) u(1) 0];
