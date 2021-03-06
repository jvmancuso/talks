library(keras)

data <- dataset_mnist(path = "mnist.npz")

X <- data$train$x/255
y <- data$train$y
X_test <- data$test$x/255
y_test <- data$test$y

mask <- runif(dim(X)[1]) < 0.8
X_train <- X[mask,,]
y_train <- y[mask]
X_val <- X[!mask,,]
y_val <- y[!mask]

y_train <- to_categorical(y_train, 10)
y_val <- to_categorical(y_val, 10)
y_test <- to_categorical(y_test, 10)

input_dim <- c(dim(X_train)[2], dim(X_train)[3])
p <- 0.2
layer_size <- 1024

ins <- layer_input(shape = input_dim)

residual_block <- function(input, size, p = 0.2){
  block <- input %>%
    layer_dense(units = size, activation = 'relu') %>%
    layer_dropout(p) %>%
    layer_batch_normalization() %>%
    layer_dense(units = size, activation = 'relu') %>%
    layer_dropout(p) %>%
    layer_batch_normalization() %>%
    layer_dense(units = size)
  result <- layer_add(c(input, block)) %>%
    activation_relu() %>%
    layer_dropout(p) %>%
    layer_batch_normalization()
  return(result)
}

outs <- ins %>% 
  layer_reshape(input_dim[1]*input_dim[2]) %>%
  layer_dense(layer_size) %>%
  residual_block(layer_size) %>%
  residual_block(layer_size) %>%
  residual_block(layer_size) %>%
  residual_block(layer_size) %>%
  residual_block(layer_size) %>%
  residual_block(layer_size) %>%
  residual_block(layer_size) %>%
  layer_dense(units = 10, activation = 'softmax')

resnet <- keras_model(inputs = ins, outputs = outs)

resnet %>% compile(
  optimizer = optimizer_adam(),
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

stopper <- callback_early_stopping(patience = 2)
checker <- callback_model_checkpoint(filepath = 'resnet_0.h5', save_best_only = TRUE)

history <- resnet %>% fit(
  X_train, y_train,
  epochs = 100, batch_size = 32,
  validation_data = list(X_val, y_val),
  callbacks = c(stopper, checker)
)

resnet <- load_model_hdf5('resnet_0.h5')

preds <- resnet %>% predict(
  X_test, batch_size = 2048
)

library(ramify)
test_acc = sum(argmax(preds) == argmax(y_test))/nrow(y_test)
print(test_acc)