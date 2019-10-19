import { HelpMessageDict } from "mlchartlib";
import { IFairnessContext } from "../IFairnessContext";
import { MetricsCache } from "../MetricsCache";

export enum FairnessChartModes {
    bar = "bar",
    beehive = "beehive",
    violin = "violin",
    box = "box"
}

export interface IFairnessChartProps {
    dashboardContext: IFairnessContext;
    messages?: HelpMessageDict;
    theme?: string;
    metricsCache: MetricsCache;
    selectedBin: number;
    selectedMetrics: string[];
    selectedModels?: number[];
    yTitle?: string;
}

export interface IFairnessChartPickerProps extends IFairnessChartProps {
    selectedChart: FairnessChartModes;
    setSelectedChart: (mode: FairnessChartModes) => void;
    metric: "accuracy" | "opportunity";
}