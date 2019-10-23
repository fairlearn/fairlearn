import { IModelMetadata } from "mlchartlib";

export interface IFairnessContext {
    // rows by [aug columns + feature columns + trueY + groupIndex]
    dataset: Array<any>;
    // modelPredictions, models x rows
    predictions: number[][];
    groupNames: string[];
    binVector: number[];
    modelMetadata: IFairnessModelMetadata;
    modelNames: string[];
}

export interface IFairnessModelMetadata extends IModelMetadata {
    predictionType: "classes" | "regression" | "probability" | "logOdds";
}