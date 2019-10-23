import React from "react";
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import { IModelComparisonProps } from "./Controls/ModelComparisonChart";
import { Text } from "office-ui-fabric-react/lib/Text";
import { localization } from "./Localization/localization";
import { Separator } from "office-ui-fabric-react/lib/Separator";
import { Spinner, SpinnerSize } from 'office-ui-fabric-react/lib/Spinner';
import { mergeStyleSets } from "@uifabric/styling";
import _ from "lodash";
import { ParityModes } from "./ParityMetrics";
import { IPlotlyProperty, AccessibleChart } from "mlchartlib";
import { ActionButton } from "office-ui-fabric-react/lib/Button";
import { SummaryTable } from "./Controls/SummaryTable";
import { PredictionTypes, IMetricResponse } from "./IFairnessProps";
import { AccuracyOptions } from "./AccuracyMetrics";
import { NONAME } from "dns";

interface IMetrics {
    globalAccuracy: number;
    accuracyDisparity: number;
    globalOutcome: number;
    outcomeDisparity: number;
    // Optional, based on model type
    binnedFNR?: number[];
    binnedFPR?: number[];
    binnedOutcome?: number[];
    binnedOverprediction?: number[];
    binnedUnderprediction?: number[];
    // different length, raw unbinned errors and predictions
    errors?: number[];
    predictions?: number[];
}

export interface IState {
    metrics?: IMetrics;
}

export interface IReportProps extends IModelComparisonProps {
    selectedModelIndex: number;
}

export class WizardReport extends React.PureComponent<IReportProps, IState> {
    private static separatorStyle = {
        root: [{
          selectors: {
            '::after': {
              backgroundColor: 'darkgrey',
            },
          }
        }]
    };

    private static readonly classNames = mergeStyleSets({
        spinner: {
            margin: "auto",
            padding: "40px"
        },
        header: {
            padding: "0 90px",
            backgroundColor: "#F2F2F2"
        },
        multimodelButton: {
            marginTop: "20px",
            padding: 0,
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "400"
        },
        headerTitle: {
            paddingTop: "10px",
            color: "#333333",
            fontSize: "32px",
            lineHeight: "39px",
            fontWeight: "100"
        },
        headerBanner: {
            display: "flex"
        },
        bannerWrapper: {
            width: "100%",
            paddingTop: "18px",
            paddingBottom: "15px",
            display: "inline-flex",
            flexDirection: "row",
            justifyContent: "space-between"
        },
        editButton: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "20px",
            fontWeight: "400"
        },
        metricText: {
            color: "#333333",
            fontSize: "36px",
            lineHeight: "44px",
            fontWeight: "100",
            paddingRight: "12px"
        },
        firstMetricLabel: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "400",
            padding: "0 17px 0 12px",
            alignSelf: "center",
            maxWidth: "90px",
            borderRight: "1px solid #CCCCCC",
            marginRight: "20px"
        },
        metricLabel: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "400",
            alignSelf: "center",
            maxWidth: "130px"
        }
    });

    private static barPlotlyProps: IPlotlyProperty = {
        config: { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian', 'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'] },
        data: [
            {
                orientation: 'h',
                type: 'bar'
            }
        ],
        layout: {
            autosize: true,
            barmode: 'relative',
            font: {
                size: 10
            },
            margin: {
                t: 10,
                l: 10
            },
            showlegend: false,
            hovermode: 'closest',
            xaxis: {
                automargin: true
            },
            yaxis: {
                automargin: true,
                showticklabels: false
            },
        } as any
    };

    render(): React.ReactNode {
        if (!this.state || !this.state.metrics) {
            this.loadData();
            return (
                <Spinner className={WizardReport.classNames.spinner} size={SpinnerSize.large} label={localization.calculating}/>
            );
        }

        const accuracyPlot = _.cloneDeep(WizardReport.barPlotlyProps);
        const opportunityPlot = _.cloneDeep(WizardReport.barPlotlyProps);
        // TODO: reverse everything
        const reversedNames = this.props.dashboardContext.groupNames;
        if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification) {
            accuracyPlot.data = [
                {
                    x: this.state.metrics.binnedFPR,
                    y: reversedNames,
                    text: this.state.metrics.binnedFPR.map(num => (num as number).toFixed(3)),
                    orientation: 'h',
                    type: 'bar',
                    textposition: 'auto'
                } as any, {
                    x: this.state.metrics.binnedFNR.map(x => -1 * x),
                    y: reversedNames,
                    text: this.state.metrics.binnedFNR.map(num => (num as number).toFixed(3)),
                    orientation: 'h',
                    type: 'bar',
                    textposition: 'auto'
                }
            ];
        } if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.probability) {
            accuracyPlot.data = [
                {
                    x: this.state.metrics.binnedOverprediction,
                    y: reversedNames,
                    text: this.state.metrics.binnedOverprediction.map(num => (num as number).toFixed(3)),
                    orientation: 'h',
                    type: 'bar',
                    textposition: 'auto'
                } as any, {
                    x: this.state.metrics.binnedUnderprediction.map(x => -1 * x),
                    y: reversedNames,
                    text: this.state.metrics.binnedUnderprediction.map(num => (num as number).toFixed(3)),
                    orientation: 'h',
                    type: 'bar',
                    textposition: 'auto'
                }
            ];
        } if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.regression) {
            accuracyPlot.data = [
                {
                    x: this.state.metrics.errors,
                    y: this.props.dashboardContext.binVector,
                    type: 'box'
                } as any
            ];
        }

        
        opportunityPlot.data = [
            {
                x: this.state.metrics.binnedOutcome,
                y: this.props.dashboardContext.groupNames,
                text: this.state.metrics.binnedOutcome.map(num => (num as number).toFixed(3)),
                orientation: 'h',
                type: 'bar',
                textposition: 'auto'
            } as any
        ];
        const globalAccuracyString = this.formatNumbers(this.state.metrics.globalAccuracy, this.props.accuracyPickerProps.selectedAccuracyKey);
        const disparityAccuracyString = this.formatNumbers(this.state.metrics.accuracyDisparity, this.props.accuracyPickerProps.selectedAccuracyKey);
        const outcomeKey = this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification ? "selection_rate" : "average";
        const globalOutcomeString = this.formatNumbers(this.state.metrics.globalOutcome, outcomeKey);
        const disparityOutcomeString = this.formatNumbers(this.state.metrics.outcomeDisparity, outcomeKey);
        return (<div style={{height: "100%", overflowY:"auto"}}>
            <div className={WizardReport.classNames.header}>
                {this.props.modelCount > 0 &&
                    <ActionButton
                        className={WizardReport.classNames.multimodelButton}
                        iconProps={{iconName: "ChevronLeft"}}
                        onClick={this.clearModelSelection}>
                        {localization.Report.backToComparisons}
                    </ActionButton>}
                <div className={WizardReport.classNames.headerTitle}>{localization.Report.title}</div>
                <div className={WizardReport.classNames.bannerWrapper}>
                    <div className={WizardReport.classNames.headerBanner}>
                        <div className={WizardReport.classNames.metricText}>{globalAccuracyString}</div>
                        <div className={WizardReport.classNames.firstMetricLabel}>{localization.Report.globalAccuracyText}</div>
                        <div className={WizardReport.classNames.metricText}>{disparityAccuracyString}</div>
                        <div className={WizardReport.classNames.metricLabel}>{localization.Report.accuracyDisparityText}</div>
                    </div>
                    <ActionButton
                        className={WizardReport.classNames.editButton}
                        iconProps={{iconName: "Edit"}}
                        onClick={this.props.onEditConfigs}>{localization.Report.editConfiguration}</ActionButton>
                </div>
            </div>
            <div style={{padding: "10px 30px", height:"400px"}}>
                <Stack horizontal styles={{root: {height: "100%"}}}>
                    <SummaryTable 
                        binLabels={this.props.dashboardContext.groupNames}
                        binValues={this.state.metrics.binnedFPR}/>
                    <Stack.Item styles={{root: {width: "55%"}}}>
                        <AccessibleChart
                            plotlyProps={accuracyPlot}
                            sharedSelectionContext={undefined}
                            theme={undefined}
                        />
                    </Stack.Item>
                </Stack>
            </div>
            <div className={WizardReport.classNames.header}>
                <div className={WizardReport.classNames.headerTitle}>{localization.Report.outcomesTitle}</div>
                <div className={WizardReport.classNames.bannerWrapper}>
                    <div className={WizardReport.classNames.headerBanner}>
                        <div className={WizardReport.classNames.metricText}>{globalOutcomeString}</div>
                        <div className={WizardReport.classNames.firstMetricLabel}>{localization.Report.globalOutcomeText}</div>
                        <div className={WizardReport.classNames.metricText}>{disparityOutcomeString}</div>
                        <div className={WizardReport.classNames.metricLabel}>{localization.Report.outcomeDisparityText}</div>
                    </div>
                </div>
            </div>
            <div style={{padding: "10px 30px", height:"400px"}}>
                <Stack horizontal styles={{root: {height: "100%"}}}>
                    <SummaryTable 
                        binLabels={this.props.dashboardContext.groupNames}
                        binValues={this.state.metrics.binnedFPR}/>
                    <Stack.Item styles={{root: {width: "55%"}}}>
                        <AccessibleChart
                            plotlyProps={opportunityPlot}
                            sharedSelectionContext={undefined}
                            theme={undefined}
                        />
                    </Stack.Item>
                </Stack>
            </div>
        </div>);
    }

    private readonly formatNumbers = (value: number, key: string, isRatio: boolean = false): string => {
        if (value === null || value === undefined) {
            return NaN.toString();
        }
        const styleObject = {maximumSignificantDigits: 3};
        if (AccuracyOptions[key].isPercentage && !isRatio) {
            (styleObject as any).style = "percent";
        }
        return value.toLocaleString(undefined, styleObject);
    }

    private readonly clearModelSelection = (): void => {
        this.props.selections.onSelect([]);
    }

    private async loadData(): Promise<void> {
        try {
            let binnedFNR: number[];
            let binnedFPR: number[];
            let binnedOverprediction: number[];
            let binnedUnderprediction: number[];
            let predictions: number[];
            let errors: number[];
            let outcomes: IMetricResponse;
            let outcomeDisparity: number;
            const globalAccuracy = (await this.props.metricsCache.getMetric(
                this.props.dashboardContext.binVector,
                this.props.featureBinPickerProps.selectedBinIndex, 
                this.props.selectedModelIndex,
                this.props.accuracyPickerProps.selectedAccuracyKey)).global;
            const accuracyDisparity = await this.props.metricsCache.getDisparityMetric(
                this.props.dashboardContext.binVector,
                this.props.featureBinPickerProps.selectedBinIndex, 
                this.props.selectedModelIndex,
                this.props.accuracyPickerProps.selectedAccuracyKey,
                ParityModes.difference);
            if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification) {
                binnedFNR = (await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "miss_rate")).bins;
                binnedFPR = (await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "fallout_rate")).bins;
                outcomes = await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "selection_rate");
                outcomeDisparity = await this.props.metricsCache.getDisparityMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "selection_rate",
                    ParityModes.difference);
            } if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.probability) {
                predictions = this.props.dashboardContext.predictions[this.props.selectedModelIndex];
                binnedOverprediction = (await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "overprediction")).bins;
                binnedUnderprediction = (await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "underprediction")).bins;
                outcomes = await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "average");
                outcomeDisparity = await this.props.metricsCache.getDisparityMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "average",
                    ParityModes.difference);
            } if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.regression) {
                predictions = this.props.dashboardContext.predictions[this.props.selectedModelIndex];
                errors = predictions.map((predicted, index) => {
                    return this.props.dashboardContext.trueY[index] - predicted;
                });
                outcomes = await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "average");
                outcomeDisparity = await this.props.metricsCache.getDisparityMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "average",
                    ParityModes.difference);
            }
            this.setState({
                metrics: {
                    globalAccuracy,
                    accuracyDisparity,
                    binnedFNR,
                    binnedFPR,
                    globalOutcome: outcomes.global,
                    binnedOutcome: outcomes.bins,
                    outcomeDisparity,
                    predictions,
                    errors,
                    binnedOverprediction,
                    binnedUnderprediction
                }
            });
        } catch {
            // todo;
        }
    }
}