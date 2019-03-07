function M = displayHookeLawMatrix(C)

M = zeros(6);

M(1,1) = C(1,1,1,1);
M(2,1) = C(2,2,1,1);
M(3,1) = C(3,3,1,1);
M(4,1) = C(2,3,1,1);
M(5,1) = C(1,3,1,1);
M(6,1) = C(1,2,1,1);

M(1,2) = C(1,1,2,2);
M(2,2) = C(2,2,2,2);
M(3,2) = C(3,3,2,2);
M(4,2) = C(2,3,2,2);
M(5,2) = C(1,3,2,2);
M(6,2) = C(1,2,2,2);

M(1,3) = C(1,1,3,3);
M(2,3) = C(2,2,3,3);
M(3,3) = C(3,3,3,3);
M(4,3) = C(2,3,3,3);
M(5,3) = C(1,3,3,3);
M(6,3) = C(1,2,3,3);

M(1,4) = C(1,1,2,3);
M(2,4) = C(2,2,2,3);
M(3,4) = C(3,3,2,3);
M(4,4) = C(2,3,2,3);
M(5,4) = C(1,3,2,3);
M(6,4) = C(1,2,2,3);

M(1,5) = C(1,1,1,3);
M(2,5) = C(2,2,1,3);
M(3,5) = C(3,3,1,3);
M(4,5) = C(2,3,1,3);
M(5,5) = C(1,3,1,3);
M(6,5) = C(1,3,1,2);
M(6,6) = C(1,2,1,2);