export const precomputedBinary = {
    trueY: [1,0,1,1,0,1,0, 0],
    predictedYs: [
        [1,0,0,1,1,1,1,0],
        [1,0,1,1,1,0,0,0],
    ],
    precomputedMetrics: [[
    // Model 1 metrics
    {
        "accuracy_score":{
            global: 0.5,
            bins: [0.2, 0.8]
        },
        "selection_rate": {
            global: 0.5,
            bins: [0.2, 0.8]
        },
        "underprediction": {
            global: 0.5,
            bins: [0.2, 0.8]
        },
        "overprediction": {
            global: 0.5,
            bins: [0.2, 0.8]
        }
    }, 
    // Model 2 metrics
    {
        "accuracy_score":{
            global: 0.6,
            bins: [0.2, 0.8]
        },
        "selection_rate": {
            global: 0.5,
            bins: [0.2, 0.8]
        },
        "underprediction": {
            global: 0.5,
            bins: [0.2, 0.8]
        },
        "overprediction": {
            global: 0.5,
            bins: [0.2, 0.8]
        }
    }],
    // Feature 2, serious real feature
    [
        // Model 1 metrics
        {
            "accuracy_score":{
                global: 0.5,
                bins: [0.2, 0.8, 0.44]
            },
            "selection_rate": {
                global: 0.5,
                bins: [0.2, 0.8, 0.44]
            },
            "underprediction": {
                global: 0.5,
                bins: [0.2, 0.8, 0.44]
            },
            "overprediction": {
                global: 0.5,
                bins: [0.2, 0.8, 0.44]
            }
        }, 
        // Model 2 metrics
        {
            "accuracy_score":{
                global: 0.6,
                bins: [0.2, 0.8, 0.44]
            },
            "selection_rate": {
                global: 0.5,
                bins: [0.2, 0.8, 0.44]
            },
            "underprediction": {
                global: 0.5,
                bins: [0.2, 0.8, 0.44]
            },
            "overprediction": {
                global: 0.5,
                bins: [0.2, 0.8, 0.44]
            }
        }]
    ],
    precomputedBins: [
        {
            binVector: [1,0,1,1,0,1,0,0],
            binLabels: ["thing 1", "thing 2"],
            featureBinName: "thingfulness"
        },
        {
            binVector: [1,2,0,1,2,0,1,2],
            binLabels: ["State A", "State B", "State C"],
            featureBinName: "serious real feature"
        }
    ],
    predictionType: "binaryClassification"
}