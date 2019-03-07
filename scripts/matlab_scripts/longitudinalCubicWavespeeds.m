function [phi, theta, v] = longitudinalCubicWavespeeds(c11, c12, c44, rho)
%Returns Longitudinal Cubic Wavespeeds as a function of two angles
%  phi is the angle the vector makes with the x1 direction in the x1-x2 
%  plane and theta is the angle between the vector and x3

% Following Miaki's note on Acoustic wave speeds and Zheng & Spencer 1993:
% c11 = lambda + 2*mu + eta
% c12 = lambda
% c13 = mu
mu = c44;
lambda = c12;
eta = c11 - lambda - 2*mu; 

% Generate 0..90 degree grids
%[theta, phi] = meshgrid(linspace(0, 180), linspace(0, 360));
[theta, phi] = meshgrid(linspace(0, 90, 501), linspace(0, 90, 501));

cT4 = cosd(theta).^4;
cP4 = cosd(phi).^4;
sP4 = sind(phi).^4;
sT4 = sind(theta).^4;
rhov2 = lambda + 2*mu + eta*(cP4.*sT4 + sP4.*sT4 + cT4); 

v = sqrt(rhov2/rho)*1.e-3; % km/s
contourf(phi, theta, v, 20)
colorbar()
xlabel('\phi','fontsize',20)
ylabel('\vartheta','fontsize',20)

hold on;
maxV = max(max(v));
[ix, iy] = find(v==maxV);
phiMax = phi(ix, iy);
thetaMax = theta(ix, iy);
plot(phiMax, thetaMax, '-p', 'MarkerFaceColor', 'Black', 'MarkerSize', 15)
text(phiMax-6, thetaMax-4, sprintf("(%2.2f, %2.2f)", phiMax, thetaMax) )
text(phiMax-6, thetaMax+4, sprintf("%2.2f km/s", maxV) )
end

