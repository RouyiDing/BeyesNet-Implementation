# BeyesNet-Implementation

 In this program, I implemented both Naive Bayes and TAN (tree-augmented Naive Bayes) to classify data.
 
 
 For the TAN algorithm:
 1. I Used Prims's algorithm to find a maximal spanning tree (but choose maximal weight edges instead of minimal weight ones). 
 2. Initialize this process by choosing the first attribute in the input file for Vnew. 
    If there are ties in selecting maximum weight edges, use the following preference criteria: 
	  (i) prefer edges emanating from attributes listed earlier in the input file;
	  (ii) if there are multiple maximal weight edges emanating from the first such attribute, prefer edges going to attributes listed earlier in the input file.
 3. To root the maximal weight spanning tree, pick the first attribute in the input file as the root.
 
 
 
 Execute:
 1. The program can be called bayes and should accept four command-line arguments as follows:
         bayes <train-set-file> <test-set-file> <n|t>
    where the last argument is a single character (either 'n' or 't') that indicates whether to use naive Bayes or TAN.
	
	
	
 Interpret output:
 1. The structure of the Bayes net by listing one line per attribute in which you list 
    (i) The name of the attribute; 
	(ii) The names of its parents in the Bayes net (for naive Bayes, this will simply be the 'class' variable for each attribute) separated by whitespace.

 2. One line for each instance in the test-set (in the same order as this file) indicating 
    (i) The predicted class; 
	(ii) The actual class;
	(iii) The posterior probability of the predicted class.
    (iv) The number of the test-set examples that were correctly classified.