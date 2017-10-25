library(keras)

data <- dataset_mnist(path = "mnist.npz")

X <- data$train$x
y <- data$train$y
X_test <- data$test$x
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
p <- 0.1
layer_size <- 1024
growth_rate <- 1/2

ins <- layer_input(shape = input_dim)

outs <- ins %>% 
  layer_reshape(c(input_dim[1]*input_dim[2])) %>%
  layer_dense(units = layer_size, kernel_initializer = 'lecun_normal', activation = 'selu') %>%
  layer_alpha_dropout(p) %>%
  layer_dense(units = layer_size, kernel_initializer = 'lecun_normal', activation = 'selu') %>%
  layer_alpha_dropout(p) %>%
  layer_dense(units = layer_size*growth_rate, kernel_initializer = 'lecun_normal', activation = 'selu') %>%
  layer_alpha_dropout(p) %>%
  layer_dense(units = layer_size*growth_rate, kernel_initializer = 'lecun_normal', activation = 'selu') %>%
  layer_alpha_dropout(p) %>%
  layer_dense(units = layer_size*(growth_rate^2), kernel_initializer = 'lecun_normal', activation = 'selu') %>%
  layer_alpha_dropout(p) %>%
  layer_dense(units = layer_size*(growth_rate^2), kernel_initializer = 'lecun_normal', activation = 'selu') %>%
  layer_alpha_dropout(p) %>%
  layer_dense(units = layer_size*(growth_rate^3), kernel_initializer = 'lecun_normal', activation = 'selu') %>%
  layer_alpha_dropout(p) %>%
  layer_dense(units = layer_size*(growth_rate^3), kernel_initializer = 'lecun_normal', activation = 'selu') %>%
  layer_alpha_dropout(p) %>%
  layer_dense(units = layer_size*(growth_rate^4), kernel_initializer = 'lecun_normal', activation = 'selu') %>%
  layer_alpha_dropout(p) %>%
  layer_dense(units = layer_size*(growth_rate^4), kernel_initializer = 'lecun_normal', activation = 'selu') %>%
  layer_alpha_dropout(p) %>%
  layer_dense(units = 10, activation = 'softmax')

selunet <- keras_model(inputs = ins, outputs = outs)

selunet %>% compile(
  optimizer = optimizer_adam(lr = 0.0001),
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

stopper <- callback_early_stopping(patience = 2)
checker <- callback_model_checkpoint(filepath = 'selunet_1.h5', save_best_only = TRUE)

history <- selunet %>% fit(
  X_train, y_train,
  epochs = 100, batch_size = 32,
  validation_data = list(X_val, y_val),
  callbacks = c(stopper, checker)
)

selunet <- load_model_hdf5('selunet_1.h5')

preds <- selunet %>% predict(
  X_test, batch_size = 2048
)

library(ramify)
test_acc = sum(argmax(preds) == argmax(y_test))/nrow(y_test)
print(test_acc)