function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% create a distance matrix
dist = zeros(rows(X), rows(centroids));

for i = 1:rows(centroids)
	D = bsxfun(@minus, X, centroids(i, :));
	dist(:, i) = sum(D .^ 2, 2);
end

[~, idx] = min(dist, [], 2);

%m = size(X,1);
%for i = 1:m
%	dist = norm(X(i) - centroids(1))^2;
%	printf("distance in %d is: %d \n", i, dist);
i%	idealCen = 1;
%	for j = 2:rows(centroids)
%		thisDist = norm(X(i) - centroids(j))^2;
%		printf("distance in %d (this is j) is: %d \n", j, thisDist);
%		if thisDist < dist
%			dist = thisDist;
%			idealCen = j;
%		endif
%	endfor
%	idx(i) = idealCen;
%endfor
% =============================================================

end

