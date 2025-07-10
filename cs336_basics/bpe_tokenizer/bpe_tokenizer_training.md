- How does the process looks like 
- First we split the training copus into pre tokens based on regex 
- i.e. "Hello! World" is ["Hello", "!", " ", "World"]
- The count of each pre token is done 
- Then these pre tokens are converted to their bytes
- And then we count the byte pairs and merge the one with highest count 


