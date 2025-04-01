clc
clear all
close all

% Incarcarea setului de date
data = readtable('spam.csv');
labels = data.v1;
messages = data.v2;

% Preprocesarea datelor
processed_messages = lower(regexprep(messages, '[^\w\s]', ''));

% Extrage caracteristici din mesaje
features = cellfun(@(x) numel(strsplit(x)), processed_messages);

% Normalizarea caracteristicilor
min_value = min(features);
max_value = max(features);
normalized_features = (features - min_value) / (max_value - min_value);

% Înlocuirea caracteristicilor inițiale cu cele normalizate
features = normalized_features;

% Etichetele ham/spam
binary_labels = strcmp(labels, 'spam');

% Împărțirea datelor în set de antrenament și de testare
train_ratio = 0.8;
num_train = round(train_ratio * numel(binary_labels));

train_features = features(1:num_train);
train_labels = binary_labels(1:num_train);

test_features = features(num_train+1:end);
test_labels = binary_labels(num_train+1:end);

%% Undersampling - Reducem numărul de exemple din clasa majoritară

% Determinăm numărul minim de exemple din cele două clase
min_class_count = min(sum(binary_labels), sum(~binary_labels));

% Selectăm aleator un subset din clasa majoritară
majority_indices = find(~binary_labels);
undersampled_majority_indices = datasample(majority_indices, min_class_count, 'Replace', false);

% Concatenăm indicii exemplelor din clasa majoritară cu cei din clasa minoritară
under_sampled_indices = [find(binary_labels); undersampled_majority_indices];

% Eliminăm valorile care depășesc dimensiunea validă
valid_indices = under_sampled_indices <= numel(train_features);
under_sampled_indices = under_sampled_indices(valid_indices);

%% Actualizăm seturile de antrenament cu exemplul sub-echilibrat
train_features = train_features(under_sampled_indices);
train_labels = train_labels(under_sampled_indices);

% Actualizăm numărul de exemple de antrenament
num_train = numel(train_labels);

%% Metoda Newton

% Inițializarea parametrilor
theta_newton = randn(2, 1) * 0.01;  % Parametrii theta0 și theta1
max_iterations = 1000;  % Numărul maxim de iterații
epsilon = 1e-6;  % Precizia pentru criteriul de oprire

% Ponderi pentru clasele spam și non-spam
w_spam = 2; % Pondere pentru clasa spam
w_non_spam = 1; % Pondere pentru clasa non-spam

% Antrenarea modelului folosind metoda Newton
tic;
for i = 1:max_iterations
    % Calculăm matricea Hessiană
    hessian = zeros(2, 2);
    for j = 1:num_train
        feature = train_features(j);
        hessian(1, 1) = hessian(1, 1) + w_non_spam;
        hessian(1, 2) = hessian(1, 2) + feature * w_non_spam;
        hessian(2, 1) = hessian(2, 1) + feature * w_non_spam;
        hessian(2, 2) = hessian(2, 2) + feature^2 * w_non_spam;
    end

    % Calculăm gradientul și eroarea
    gradient = zeros(2, 1);
    error = 0;
    for j = 1:num_train
        feature = train_features(j);
        label = train_labels(j);
        predicted = theta_newton(1) + theta_newton(2) * feature;
        error = error + w_spam * (predicted - label)^2 + w_non_spam * (predicted - label)^2;
        gradient(1) = gradient(1) + w_spam * (predicted - label);
        gradient(2) = gradient(2) + w_non_spam * (predicted - label) * feature;
    end

    % Actualizăm parametrii folosind metoda Newton
    delta_theta = hessian \ gradient;
    theta_newton = theta_newton - delta_theta;

    % Verificăm criteriul de oprire
    if norm(delta_theta) < epsilon
        break;
    end
end
training_time_newton = toc;

%% Evaluarea modelului pe setul de testare (metoda Newton)
tic;
test_predictions_newton = theta_newton(1) + theta_newton(2) * test_features;
test_predictions_binary_newton = test_predictions_newton >= 0.5;
accuracy_newton = sum(test_predictions_binary_newton == test_labels) / numel(test_labels) * 100;
evaluation_time_newton = toc;

% Afișarea rezultatelor
disp("METODA NEWTON:");
disp(' ');
disp("Timp de antrenare: " + training_time_newton + " secunde");
disp("Timp de evaluare: " + evaluation_time_newton + " secunde");
disp(' ');

% Calculați matricea de confuzie pentru metoda Newton
confusion_matrix_newton = zeros(2, 2);
for i = 1:numel(test_labels)
    true_label = test_labels(i);
    predicted_label = test_predictions_binary_newton(i);
    confusion_matrix_newton(true_label + 1, predicted_label + 1) = confusion_matrix_newton(true_label + 1, predicted_label + 1) + 1;
end

% Afișați matricea de confuzie pentru metoda Newton
disp("Matricea de confuzie (metoda Newton):");
disp(confusion_matrix_newton);
disp(' ');


disp("Precizia modelului pe setul de testare (metoda Newton): " + accuracy_newton + "%");

% Grafic etichete reale versus predicții (metoda Newton)
figure;
scatter(1:numel(test_labels), test_labels, 'ro', 'filled');
hold on;
plot(1:numel(test_labels), test_predictions_binary_newton, 'bx', 'LineWidth', 1);
xlabel('Exemplu');
ylabel('Etichetă (0 - ham, 1 - spam)');
title('Etichete reale versus predicții (metoda Newton)');
legend('Etichete reale', 'Predicții');
grid on;
xlim([0, numel(test_labels)+1]);
ylim([-0.5, 1.5]);

disp(' ');
disp("------------------------------------------");
disp(' ');

%% Metoda Gradient Stocastic

% Inițializarea parametrilor
theta_stochastic = randn(2, 1) * 0.01;  % Parametrii theta0 și theta1
max_iterations = 1000;  % Numărul maxim de iterații
epsilon = 1e-6;  % Precizia pentru criteriul de oprire
learning_rate = 0.011;  % Rata de învățare

% Ponderi pentru clasele spam și non-spam
w_spam = 2; % Pondere pentru clasa spam
w_non_spam = 1; % Pondere pentru clasa non-spam

% Antrenarea modelului folosind metoda Gradientului Stocastic
tic;
for i = 1:max_iterations
    % Selectăm un exemplu aleator din setul de antrenament
    random_index = randi(num_train);
    feature = train_features(random_index);
    label = train_labels(random_index);
    
    % Calculăm predicția pentru exemplul selectat
    predicted = theta_stochastic(1) + theta_stochastic(2) * feature;
    
    % Calculăm gradientul și eroarea pentru exemplul selectat
    error = w_spam * (predicted - label)^2 + w_non_spam * (predicted - label)^2;
    gradient(1) = w_spam * (predicted - label);
    gradient(2) = w_non_spam * (predicted - label) * feature;
    
    % Actualizăm parametrii folosind metoda Gradientului Stocastic
    theta_stochastic = theta_stochastic - learning_rate * gradient;
     
    % Verificăm criteriul de oprire
    if norm(learning_rate * gradient) < epsilon
        break;
    end
end
training_time_stochastic = toc;


%% Evaluarea modelului pe setul de testare (metoda Gradient Stocastic)
tic;
test_predictions_stochastic = theta_stochastic(1) + theta_stochastic(2) * test_features;
test_predictions_binary_stochastic = test_predictions_stochastic >= 0.5;
accuracy_stochastic = sum(test_predictions_binary_stochastic == test_labels) / numel(test_labels) * 100;
evaluation_time_stochastic = toc;

% Afișarea rezultatelor
disp("METODA GRADIENT STOCASTIC:");
disp(' ');
disp("Timp de antrenare: " + training_time_stochastic + " secunde");
disp("Timp de evaluare: " + evaluation_time_stochastic + " secunde");
disp(' ');

% Calculați matricea de confuzie pentru metoda Gradient Stocastic
confusion_matrix_stochastic = zeros(2, 2);
for i = 1:numel(test_labels)
    true_label = test_labels(i);
    predicted_label = test_predictions_binary_stochastic(i);
    confusion_matrix_stochastic(true_label + 1, predicted_label + 1) = confusion_matrix_stochastic(true_label + 1, predicted_label + 1) + 1;
end

% Afișați matricea de confuzie pentru metoda Gradient Stocastic
disp("Matricea de confuzie (metoda Gradient Stocastic):");
disp(confusion_matrix_stochastic);
disp(' ');

disp("Precizia modelului pe setul de testare (metoda Gradient Stocastic): " + accuracy_stochastic + "%");

% Grafic etichete reale versus predicții (metoda Gradient Stocastic)
figure;
scatter(1:numel(test_labels), test_labels, 'ro', 'filled');
hold on;
plot(1:numel(test_labels), test_predictions_binary_stochastic, 'bx', 'LineWidth', 1);
xlabel('Exemplu');
ylabel('Etichetă (0 - ham, 1 - spam)');
title('Etichete reale versus predicții (metoda Gradient stocastică)');
legend('Etichete reale', 'Predicții');
grid on;
xlim([0, numel(test_labels)+1]);
ylim([-0.5, 1.5]);

disp(' ');
disp("------------------------------------------");
disp(' ');

%% Metoda fminunc

% Inițializarea parametrilor
theta_stochastic = randn(2, 1) * 0.01;  % Parametrii theta0 și theta1
max_iterations = 1000;  % Numărul maxim de iterații
epsilon = 1e-6;  % Precizia pentru criteriul de oprire
learning_rate = 0.011;  % Rata de învățare

% Ponderi pentru clasele spam și non-spam
w_spam = 2; % Pondere pentru clasa spam
w_non_spam = 1; % Pondere pentru clasa non-spam

% Definirea funcției de cost
cost_function = @(theta) compute_cost(theta, train_features, train_labels, w_spam, w_non_spam);

% Inițializarea parametrilor
theta_initial = randn(2, 1) * 0.01;  % Parametrii theta0 și theta1

% Specificarea opțiunilor pentru metoda de optimizare
options = optimset('MaxIter', max_iterations, 'TolX', epsilon);

% Aplicarea metodei de optimizare pentru minimizarea funcției de cost
tic;
theta_optimized = fminunc(cost_function, theta_initial, options);
training_time = toc;

% Evaluarea modelului pe setul de testare
tic;
test_predictions = theta_optimized(1) + theta_optimized(2) * test_features;
test_predictions_binary = test_predictions >= 0.5;
accuracy = sum(test_predictions_binary == test_labels) / numel(test_labels) * 100;
evaluation_time = toc;

% Afișarea rezultatelor
disp("METODA FMINUNC:");
disp(' ');
disp("Timp de antrenare: " + training_time + " secunde");
disp("Timp de evaluare: " + evaluation_time + " secunde");
disp(' ');

% Calculați matricea de confuzie pentru metoda de optimizare
confusion_matrix = zeros(2, 2);
for i = 1:numel(test_labels)
    true_label = test_labels(i);
    predicted_label = test_predictions_binary(i);
    confusion_matrix(true_label + 1, predicted_label + 1) = confusion_matrix(true_label + 1, predicted_label + 1) + 1;
end

% Afișați matricea de confuzie pentru metoda de optimizare
disp("Matricea de confuzie (metoda FMINUNC:");
disp(confusion_matrix);
disp(' ');

disp("Precizia modelului pe setul de testare (metoda FMINUNC): " + accuracy + "%");

% Grafic etichete reale versus predicții (metoda de optimizare)
figure;
scatter(1:numel(test_labels), test_labels, 'ro', 'filled');
hold on;
plot(1:numel(test_labels), test_predictions_binary, 'bx', 'LineWidth', 1);
xlabel('Exemplu');
ylabel('Etichetă (0 - ham, 1 - spam)');
title('Etichete reale versus predicții (metoda FMINUNC)');
legend('Etichete reale', 'Predicții');
grid on;
xlim([0, numel(test_labels)+1]);
ylim([-0.5, 1.5]);

disp(' ');
disp("------------------------------------------");
disp(' ');

%% Metoda CVX

% Formularea și rezolvarea problemei de optimizare
tic;
cvx_solver sdpt3
cvx_begin
    variable theta_cvx(2)  % Parametrii theta0 și theta1
    minimize(sum(w_spam * (theta_cvx(1) + theta_cvx(2) * train_features - train_labels).^2) + ...
             sum(w_non_spam * (theta_cvx(1) + theta_cvx(2) * train_features - train_labels).^2))
cvx_end
evaluation_time_newton = toc;

%% Evaluarea modelului pe setul de testare (metoda CVX)
test_predictions_cvx = theta_cvx(1) + theta_cvx(2) * test_features;
test_predictions_binary_cvx = test_predictions_cvx >= 0.5;
accuracy_cvx = sum(test_predictions_binary_cvx == test_labels) / numel(test_labels) * 100;
evaluation_time_newton = toc;

% Afișarea rezultatelor
disp("METODA CVX:");
disp(' ');
disp("Timp de antrenare: " + training_time_newton + " secunde");
disp("Timp de evaluare: " + evaluation_time_newton + " secunde");
disp(' ');

% Calculați matricea de confuzie pentru metoda CVX
confusion_matrix_cvx = zeros(2, 2);
for i = 1:numel(test_labels)
    true_label = test_labels(i);
    predicted_label = test_predictions_binary_cvx(i);
    confusion_matrix_cvx(true_label + 1, predicted_label + 1) = confusion_matrix_cvx(true_label + 1, predicted_label + 1) + 1;
end

% Afișați matricea de confuzie pentru metoda CVX
disp("Matricea de confuzie (metoda CVX):");
disp(confusion_matrix_cvx);
disp(' ');

disp("Precizia modelului pe setul de testare (metoda CVX): " + accuracy_cvx + "%");

% Grafic etichete reale versus predicții (metoda CVX)
figure;
scatter(1:numel(test_labels), test_labels, 'ro', 'filled');
hold on;
plot(1:numel(test_labels), test_predictions_binary_cvx, 'bx', 'LineWidth', 1);
xlabel('Exemplu');
ylabel('Etichetă (0 - ham, 1 - spam)');
title('Etichete reale versus predicții (metoda CVX)');
legend('Etichete reale', 'Predicții');
grid on;
xlim([0, numel(test_labels)+1]);
ylim([-0.5, 1.5]);


% Definirea funcției de cost pentru fminunc
function cost = compute_cost(theta, features, labels, w_spam, w_non_spam)
    num_samples = numel(labels);
    error = 0;
    for i = 1:num_samples
        feature = features(i);
        label = labels(i);
        predicted = theta(1) + theta(2) * feature;
        error = error + w_spam * (predicted - label)^2 + w_non_spam * (predicted - label)^2;
    end
    cost = error;
end