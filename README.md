<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Number Prediction Panel</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <style>
        body {
            background-color: #0a0a0a;
            color: cyan;
            font-family: Arial, sans-serif;
            text-align: center;
            margin-top: 50px;
        }

        .container {
            max-width: 600px;
            margin: auto;
            padding: 20px;
            background: #111;
            border-radius: 10px;
            box-shadow: 0 0 15px cyan;
        }

        h1 {
            margin-bottom: 20px;
        }

        .prediction-box {
            font-size: 28px;
            font-weight: bold;
            margin: 20px 0;
            color: #00ff00;
        }

        button {
            padding: 12px 24px;
            font-size: 18px;
            margin-top: 20px;
            cursor: pointer;
            background-color: #00ff00;
            color: #111;
            border: none;
            border-radius: 8px;
        }

        button:hover {
            background-color: #00cc00;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>AI Number Prediction Panel (0-9)</h1>
        <div id="prediction" class="prediction-box">Waiting for Prediction...</div>
        <button onclick="predictNumber()">Predict Next Number</button>
    </div>

    <script>
        // Define model variables
        let model;
        let historicalData = [
            3, 7, 1, 9, 5, 2, 0, 6, 8, 4, 3, 5, 7, 8, 1, 4, 0, 2, 9, 6
        ]; // Example past results to train

        // Prepare Data for Training
        function prepareData() {
            const xs = [];
            const ys = [];
            for (let i = 0; i < historicalData.length - 1; i++) {
                xs.push([historicalData[i]]);
                ys.push(historicalData[i + 1]);
            }
            return { xs: tf.tensor2d(xs, [xs.length, 1]), ys: tf.tensor2d(ys, [ys.length, 1]) };
        }

        // Train Model
        async function trainModel() {
            const data = prepareData();

            model = tf.sequential();
            model.add(tf.layers.dense({ units: 16, inputShape: [1], activation: 'relu' }));
            model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
            model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

            model.compile({ optimizer: 'adam', loss: 'sparseCategoricalCrossentropy', metrics: ['accuracy'] });

            await model.fit(data.xs, data.ys, { epochs: 100 });
            console.log('✅ Model training complete.');
        }

        // Predict Next Number
        async function predictNumber() {
            if (!model) {
                document.getElementById('prediction').innerText = '⚙️ Training Model...';
                await trainModel();
            }

            const lastResult = historicalData[historicalData.length - 1];
            const prediction = model.predict(tf.tensor2d([[lastResult]], [1, 1]));
            const predictedValue = prediction.argMax(-1).dataSync()[0];

            document.getElementById('prediction').innerText = `🎯 Predicted Number: ${predictedValue}`;
            historicalData.push(predictedValue);

            // Limit historical data to 20 entries
            if (historicalData.length > 20) {
                historicalData.shift();
            }
        }

        // Auto Train on Load
        window.onload = async () => {
            document.getElementById('prediction').innerText = '⚙️ Training Model...';
            await trainModel();
        };
    </script>
</body>

</html>
