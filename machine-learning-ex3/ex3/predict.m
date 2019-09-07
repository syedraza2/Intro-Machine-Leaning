function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

% gives a2 25x5000
a2=sigmoid(Theta1*X');
%Add row of ones
a2n=[ones(1,m);a2];

%gives 10x26  26x5000
a3=sigmoid(Theta2*a2n);

%a3 is now 10X5000

%old code for LR
%this gives labels x m = 10x5000
%[maxval,p]=max(plarge, [], 1);

%pixks out index of maximum value in row
[maxval,p]=max(a3, [], 1);


% =========================================================================


end
