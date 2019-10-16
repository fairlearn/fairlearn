import { IMetricResponse, IMetricRequest } from "./IFairnessProps";
import { ParityModes } from "./ParityMetrics";

export class MetricsCache {
    private static readonly defaultKeys = ["falsePositiveRate", "falseNegativeRate"]; 

    // Top index is featureBin index, second index is model index. Third string key is metricKey
    private cache: Array<Array<{[key: string]: IMetricResponse}>>;
    constructor(featureCount: number,
        numberOfModels: number,
        private fetchMethod: (request: IMetricRequest) =>  Promise<IMetricResponse>) {
        this.cache = new Array(featureCount).fill(0).map(y => new Array(numberOfModels).fill(0).map(x => {return {};}));
    }

    public async getMetric(binIndexVector: number[], featureIndex: number, modelIndex: number, key: string): Promise<IMetricResponse> {
        let value = this.cache[featureIndex][modelIndex][key];
        if (value === undefined) {
            value = await this.fetchMethod({
                metricKey:key,
                binVector: binIndexVector,
                modelIndex: modelIndex
            });
            this.cache[featureIndex][modelIndex][key] = value;
        }
        return this.cache[featureIndex][modelIndex][key];
    }

    public async getDisparityMetric(binIndexVector: number[], featureIndex: number, modelIndex: number, key: string, disparityMethod: ParityModes): Promise<number> {
        let value = this.cache[featureIndex][modelIndex][key];
        if (value === undefined) {
            value = await this.fetchMethod({
                metricKey: key,
                binVector: binIndexVector,
                modelIndex: modelIndex
            });
            this.cache[featureIndex][modelIndex][key] = value;
        }

        const min = Math.min(...(value.bins as number[]));
        const max = Math.max(...(value.bins as number[]));
        if (isNaN(min) || isNaN(max) || (max === 0 && disparityMethod === ParityModes.ratio)) {
            return Number.NaN;
        }
        return disparityMethod === ParityModes.difference ?
            max - min :
            min / max;
    }
}