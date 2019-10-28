import { IModelMetadata } from "mlchartlib";
import { PredictionType } from "./IFairnessProps";

export interface IFairnessContext {
    // rows by [aug columns + feature columns + trueY + groupIndex]
    dataset: Array<any>;
    trueY: number[];
    // modelPredictions, models x rows
    predictions: number[][];
    groupNames: string[];
    binVector: number[];
    modelMetadata: IFairnessModelMetadata;
    modelNames: string[];
}

export interface IFairnessModelMetadata extends IModelMetadata {
    predictionType: PredictionType;
}