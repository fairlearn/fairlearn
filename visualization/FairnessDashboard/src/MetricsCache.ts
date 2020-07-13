import { IMetricResponse, IMetricRequest } from './IFairnessProps';
import { ParityModes } from './ParityMetrics';

export class MetricsCache {
    private static readonly defaultKeys = ['falsePositiveRate', 'falseNegativeRate'];

    // Top index is featureBin index, second index is model index. Third string key is metricKey
    private cache: Array<Array<{ [key: string]: IMetricResponse }>>;
    constructor(
        private featureCount: number,
        private numberOfModels: number,
        private fetchMethod: (request: IMetricRequest) => Promise<IMetricResponse>,
        precomputedCache?: Array<Array<{ [key: string]: IMetricResponse }>>,
    ) {
        if (precomputedCache) {
            this.cache = precomputedCache;
        } else {
            this.cache = new Array(featureCount).fill(0).map((y) =>
                new Array(numberOfModels).fill(0).map((x) => {
                    return {};
                }),
            );
        }
    }

    public async getMetric(
        binIndexVector: number[],
        featureIndex: number,
        modelIndex: number,
        key: string,
    ): Promise<IMetricResponse> {
        let value = this.cache[featureIndex][modelIndex][key];
        if (value === undefined && this.fetchMethod) {
            value = await this.fetchMethod({
                metricKey: key,
                binVector: binIndexVector,
                modelIndex: modelIndex,
            });
            this.cache[featureIndex][modelIndex][key] = value;
        }
        return this.cache[featureIndex][modelIndex][key];
    }

    public async getDisparityMetric(
        binIndexVector: number[],
        featureIndex: number,
        modelIndex: number,
        key: string,
        disparityMethod: ParityModes,
    ): Promise<number> {
        let value = this.cache[featureIndex][modelIndex][key];
        if (value === undefined && this.fetchMethod) {
            value = await this.fetchMethod({
                metricKey: key,
                binVector: binIndexVector,
                modelIndex: modelIndex,
            });
            this.cache[featureIndex][modelIndex][key] = value;
        }

        const bins = value.bins.slice().filter((x) => x !== undefined && !isNaN(x));

        const min = Math.min(...(bins as number[]));
        const max = Math.max(...(bins as number[]));
        if (isNaN(min) || isNaN(max) || (max === 0 && disparityMethod === ParityModes.ratio)) {
            return Number.NaN;
        }
        return disparityMethod === ParityModes.difference ? max - min : min / max;
    }

    public clearCache(binIndex?: number): void {
        if (binIndex !== undefined) {
            this.cache[binIndex] = new Array(this.numberOfModels).fill(0).map((x) => {
                return {};
            });
        } else {
            this.cache = new Array(this.featureCount).fill(0).map((y) =>
                new Array(this.numberOfModels).fill(0).map((x) => {
                    return {};
                }),
            );
        }
    }
}
