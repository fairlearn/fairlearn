import { getTheme } from '@uifabric/styling';
import _ from 'lodash';
import { AccessibleChart, IPlotlyProperty } from 'mlchartlib';
import { ActionButton } from 'office-ui-fabric-react/lib/Button';
import { Spinner, SpinnerSize } from 'office-ui-fabric-react/lib/Spinner';
import { Text } from 'office-ui-fabric-react/lib/Text';
import React from 'react';
import { AccuracyOptions } from './AccuracyMetrics';
import { ChartColors } from './ChartColors';
import { IModelComparisonProps } from './Controls/ModelComparisonChart';
import { SummaryTable } from './Controls/SummaryTable';
import { IMetricResponse, PredictionTypes } from './IFairnessProps';
import { localization } from './Localization/localization';
import { ParityModes } from './ParityMetrics';
import { WizardReportStyles } from './WizardReport.styles';

const theme = getTheme();
interface IMetrics {
    globalAccuracy: number;
    binnedAccuracy: number[];
    accuracyDisparity: number;
    globalOutcome: number;
    outcomeDisparity: number;
    binnedOutcome: number[];
    // Optional, based on model type
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
        root: [
            {
                selectors: {
                    '::after': {
                        backgroundColor: theme.semanticColors.bodyFrameBackground,
                    },
                },
            },
        ],
    };

    private static barPlotlyProps: IPlotlyProperty = {
        config: {
            displaylogo: false,
            responsive: true,
            modeBarButtonsToRemove: [
                'toggleSpikelines',
                'hoverClosestCartesian',
                'hoverCompareCartesian',
                'zoom2d',
                'pan2d',
                'select2d',
                'lasso2d',
                'zoomIn2d',
                'zoomOut2d',
                'autoScale2d',
                'resetScale2d',
            ],
        },
        data: [
            {
                orientation: 'h',
                type: 'bar',
            },
        ],
        layout: {
            autosize: true,
            barmode: 'relative',
            font: {
                size: 10,
            },
            margin: {
                t: 4,
                l: 0,
                r: 0,
                b: 20,
            },
            showlegend: false,
            hovermode: 'closest',
            plot_bgcolor: theme.semanticColors.bodyFrameBackground,
            xaxis: {
                fixedrange: true,
                autorange: true,
                mirror: true,
                linecolor: theme.semanticColors.disabledBorder,
                linewidth: 1,
            },
            yaxis: {
                fixedrange: true,
                showticklabels: false,
                showgrid: true,
                dtick: 1,
                tick0: 0.5,
                gridcolor: theme.semanticColors.disabledBorder,
                gridwidth: 1,
                autorange: 'reversed',
            },
        } as any,
    };

    render(): React.ReactNode {
        const styles = WizardReportStyles();
        if (!this.state || !this.state.metrics) {
            this.loadData();
            return <Spinner className={styles.spinner} size={SpinnerSize.large} label={localization.calculating} />;
        }

        const alternateHeight =
            this.props.featureBinPickerProps.featureBins[this.props.featureBinPickerProps.selectedBinIndex].labelArray
                .length *
                60 +
            106;
        const areaHeights = Math.max(460, alternateHeight);

        const accuracyKey = this.props.accuracyPickerProps.selectedAccuracyKey;
        const outcomeKey =
            this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification
                ? 'selection_rate'
                : 'average';
        const outcomeMetric = AccuracyOptions[outcomeKey];

        const accuracyPlot = _.cloneDeep(WizardReport.barPlotlyProps);
        const opportunityPlot = _.cloneDeep(WizardReport.barPlotlyProps);
        const nameIndex = this.props.dashboardContext.groupNames.map((unuxed, i) => i);
        let howToReadAccuracySection: React.ReactNode;
        let insightsAccuracySection: React.ReactNode;
        let howToReadOutcomesSection: React.ReactNode;
        let insightsOutcomesSection: React.ReactNode;
        let accuracyChartHeader = '';
        let opportunityChartHeader = '';

        if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification) {
            accuracyPlot.data = [
                {
                    x: this.state.metrics.binnedOverprediction,
                    y: nameIndex,
                    text: this.state.metrics.binnedOverprediction.map((num) =>
                        this.formatNumbers(num as number, 'accuracy_score', false, 2),
                    ),
                    name: localization.Metrics.overprediction,
                    width: 0.5,
                    color: ChartColors[0],
                    orientation: 'h',
                    type: 'bar',
                    textposition: 'auto',
                    hoverinfo: 'skip',
                } as any,
                {
                    x: this.state.metrics.binnedUnderprediction.map((x) => -1 * x),
                    y: nameIndex,
                    text: this.state.metrics.binnedUnderprediction.map((num) =>
                        this.formatNumbers(num as number, 'accuracy_score', false, 2),
                    ),
                    name: localization.Metrics.underprediction,
                    width: 0.5,
                    color: ChartColors[1],
                    orientation: 'h',
                    type: 'bar',
                    textposition: 'auto',
                    hoverinfo: 'skip',
                },
            ];
            accuracyPlot.layout.annotations = [
                {
                    text: localization.Report.underestimationError,
                    x: 0.02,
                    y: 1,
                    yref: 'paper',
                    xref: 'paper',
                    showarrow: false,
                    font: { color: theme.semanticColors.bodySubtext, size: 10 },
                },
                {
                    text: localization.Report.overestimationError,
                    x: 0.98,
                    y: 1,
                    yref: 'paper',
                    xref: 'paper',
                    showarrow: false,
                    font: { color: theme.semanticColors.bodySubtext, size: 10 },
                },
            ];
            accuracyPlot.layout.xaxis.tickformat = ',.0%';
            opportunityPlot.data = [
                {
                    x: this.state.metrics.binnedOutcome,
                    y: nameIndex,
                    text: this.state.metrics.binnedOutcome.map((num) =>
                        this.formatNumbers(num as number, 'selection_rate', false, 2),
                    ),
                    name: outcomeMetric.title,
                    color: ChartColors[0],
                    orientation: 'h',
                    type: 'bar',
                    textposition: 'auto',
                    hoverinfo: 'skip',
                } as any,
            ];
            opportunityPlot.layout.xaxis.tickformat = ',.0%';
            howToReadAccuracySection = (
                <div className={styles.rightText}>
                    <div className={styles.textRow}>
                        <div className={styles.colorBlock} style={{ backgroundColor: ChartColors[1] }} />
                        <div>
                            <Text block>{localization.Report.underestimationError}</Text>
                            <Text block>{localization.Report.underpredictionExplanation}</Text>
                        </div>
                    </div>
                    <div className={styles.textRow}>
                        <div className={styles.colorBlock} style={{ backgroundColor: ChartColors[0] }} />
                        <div>
                            <Text block>{localization.Report.overestimationError}</Text>
                            <Text block>{localization.Report.overpredictionExplanation}</Text>
                        </div>
                    </div>
                    <Text block>{localization.Report.classificationAccuracyHowToRead1}</Text>
                    <Text block>{localization.Report.classificationAccuracyHowToRead2}</Text>
                    <Text block>{localization.Report.classificationAccuracyHowToRead3}</Text>
                </div>
            );
            howToReadOutcomesSection = (
                <Text className={styles.textRow} block>
                    {localization.Report.classificationOutcomesHowToRead}
                </Text>
            );
        }
        if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.probability) {
            accuracyPlot.data = [
                {
                    x: this.state.metrics.binnedOverprediction,
                    y: nameIndex,
                    text: this.state.metrics.binnedOverprediction.map((num) =>
                        this.formatNumbers(num as number, 'overprediction', false, 2),
                    ),
                    name: localization.Metrics.overprediction,
                    width: 0.5,
                    color: ChartColors[0],
                    orientation: 'h',
                    type: 'bar',
                    textposition: 'auto',
                    hoverinfo: 'skip',
                } as any,
                {
                    x: this.state.metrics.binnedUnderprediction.map((x) => -1 * x),
                    y: nameIndex,
                    text: this.state.metrics.binnedUnderprediction.map((num) =>
                        this.formatNumbers(num as number, 'underprediction', false, 2),
                    ),
                    name: localization.Metrics.underprediction,
                    width: 0.5,
                    color: ChartColors[1],
                    orientation: 'h',
                    type: 'bar',
                    textposition: 'auto',
                    hoverinfo: 'skip',
                },
            ];
            accuracyPlot.layout.annotations = [
                {
                    text: localization.Report.underestimationError,
                    x: 0.1,
                    y: 1,
                    yref: 'paper',
                    xref: 'paper',
                    showarrow: false,
                    font: { color: theme.semanticColors.bodySubtext, size: 10 },
                },
                {
                    text: localization.Report.overestimationError,
                    x: 0.9,
                    y: 1,
                    yref: 'paper',
                    xref: 'paper',
                    showarrow: false,
                    font: { color: theme.semanticColors.bodySubtext, size: 10 },
                },
            ];
            const opportunityText = this.state.metrics.predictions.map((val) => {
                return localization.formatString(
                    localization.Report.tooltipPrediction,
                    this.formatNumbers(val as number, 'average', false, 3),
                );
            });
            opportunityPlot.data = [
                {
                    x: this.state.metrics.predictions,
                    y: this.props.dashboardContext.binVector,
                    text: opportunityText,
                    type: 'box',
                    color: ChartColors[0],
                    boxmean: true,
                    orientation: 'h',
                    boxpoints: 'all',
                    hoverinfo: 'text',
                    hoveron: 'points',
                    jitter: 0.4,
                    pointpos: 0,
                } as any,
            ];
            howToReadAccuracySection = (
                <div>
                    <div className={styles.textRow}>
                        <div className={styles.colorBlock} style={{ backgroundColor: ChartColors[0] }} />
                        <Text block>{localization.Report.overestimationError}</Text>
                    </div>
                    <div className={styles.textRow}>
                        <div className={styles.colorBlock} style={{ backgroundColor: ChartColors[1] }} />
                        <Text block>{localization.Report.underestimationError}</Text>
                    </div>
                    <Text className={styles.textRow} block>
                        {localization.Report.probabilityAccuracyHowToRead1}
                    </Text>
                    <Text className={styles.textRow} block>
                        {localization.Report.probabilityAccuracyHowToRead2}
                    </Text>
                    <Text className={styles.textRow} block>
                        {localization.Report.probabilityAccuracyHowToRead3}
                    </Text>
                </div>
            );
            howToReadOutcomesSection = (
                <div>
                    <Text className={styles.textRow} block>
                        {localization.Report.regressionOutcomesHowToRead}
                    </Text>
                </div>
            );
            opportunityChartHeader = localization.Report.distributionOfPredictions;
        }
        if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.regression) {
            const opportunityText = this.state.metrics.predictions.map((val) => {
                return localization.formatString(localization.Report.tooltipPrediction, val);
            });
            const accuracyText = this.state.metrics.predictions.map((val, index) => {
                return `${localization.formatString(
                    localization.Report.tooltipError,
                    this.formatNumbers(this.state.metrics.errors[index] as number, 'average', false, 3),
                )}<br>${localization.formatString(
                    localization.Report.tooltipPrediction,
                    this.formatNumbers(val as number, 'average', false, 3),
                )}`;
            });
            accuracyPlot.data = [
                {
                    x: this.state.metrics.errors,
                    y: this.props.dashboardContext.binVector,
                    text: accuracyText,
                    type: 'box',
                    color: ChartColors[0],
                    orientation: 'h',
                    boxmean: true,
                    hoveron: 'points',
                    hoverinfo: 'text',
                    boxpoints: 'all',
                    jitter: 0.4,
                    pointpos: 0,
                } as any,
            ];
            opportunityPlot.data = [
                {
                    x: this.state.metrics.predictions,
                    y: this.props.dashboardContext.binVector,
                    text: opportunityText,
                    type: 'box',
                    color: ChartColors[0],
                    boxmean: true,
                    orientation: 'h',
                    hoveron: 'points',
                    boxpoints: 'all',
                    hoverinfo: 'text',
                    jitter: 0.4,
                    pointpos: 0,
                } as any,
            ];
            howToReadAccuracySection = (
                <div>
                    <Text className={styles.textRow} block>
                        {localization.Report.regressionAccuracyHowToRead}
                    </Text>
                </div>
            );
            howToReadOutcomesSection = (
                <div>
                    <Text className={styles.textRow} block>
                        {localization.Report.regressionOutcomesHowToRead}
                    </Text>
                </div>
            );
            opportunityChartHeader = localization.Report.distributionOfPredictions;
            accuracyChartHeader = localization.Report.distributionOfErrors;
        }

        const globalAccuracyString = this.formatNumbers(this.state.metrics.globalAccuracy, accuracyKey);
        const disparityAccuracyString = this.formatNumbers(this.state.metrics.accuracyDisparity, accuracyKey);
        let selectedMetric = AccuracyOptions[this.props.accuracyPickerProps.selectedAccuracyKey];
        // handle custom metric case
        if (selectedMetric === undefined) {
            selectedMetric = this.props.accuracyPickerProps.accuracyOptions.find(
                (metric) => metric.key === this.props.accuracyPickerProps.selectedAccuracyKey,
            );
        }

        const globalOutcomeString = this.formatNumbers(this.state.metrics.globalOutcome, outcomeKey);
        const disparityOutcomeString = this.formatNumbers(this.state.metrics.outcomeDisparity, outcomeKey);
        const formattedBinAccuracyValues = this.state.metrics.binnedAccuracy.map((value) =>
            this.formatNumbers(value, accuracyKey),
        );
        const formattedBinOutcomeValues = this.state.metrics.binnedOutcome.map((value) =>
            this.formatNumbers(value, outcomeKey),
        );
        return (
            <div style={{ height: '100%', overflowY: 'auto' }}>
                <div className={styles.header}>
                    {this.props.modelCount > 1 && (
                        <div className={styles.multimodelSection}>
                            <ActionButton
                                className={styles.multimodelButton}
                                iconProps={{ iconName: 'ChevronLeft' }}
                                onClick={this.clearModelSelection}
                            >
                                {localization.Report.backToComparisons}
                            </ActionButton>
                            <Text variant={'large'} className={styles.modelLabel}>
                                {this.props.dashboardContext.modelNames[this.props.selectedModelIndex]}
                            </Text>
                        </div>
                    )}
                    <Text variant={'mediumPlus'} className={styles.headerTitle}>
                        {localization.Report.title}
                    </Text>
                    <div className={styles.bannerWrapper}>
                        <div className={styles.headerBanner}>
                            <Text className={styles.metricText} block>
                                {globalAccuracyString}
                            </Text>
                            <Text className={styles.firstMetricLabel} block>
                                {localization.formatString(
                                    localization.Report.globalAccuracyText,
                                    selectedMetric.alwaysUpperCase
                                        ? selectedMetric.title
                                        : selectedMetric.title.toLowerCase(),
                                )}
                            </Text>
                            <Text className={styles.metricText} block>
                                {disparityAccuracyString}
                            </Text>
                            <Text className={styles.metricLabel} block>
                                {localization.formatString(
                                    localization.Report.accuracyDisparityText,
                                    selectedMetric.alwaysUpperCase
                                        ? selectedMetric.title
                                        : selectedMetric.title.toLowerCase(),
                                )}
                            </Text>
                        </div>
                        <ActionButton iconProps={{ iconName: 'Edit' }} onClick={this.onEditConfigs}>
                            {localization.Report.editConfiguration}
                        </ActionButton>
                    </div>
                </div>
                <div className={styles.presentationArea} style={{ height: `${areaHeights}px` }}>
                    <SummaryTable
                        binGroup={
                            this.props.dashboardContext.modelMetadata.featureNames[
                                this.props.featureBinPickerProps.selectedBinIndex
                            ]
                        }
                        binLabels={this.props.dashboardContext.groupNames}
                        formattedBinValues={formattedBinAccuracyValues}
                        metricLabel={selectedMetric.title}
                        binValues={this.state.metrics.binnedAccuracy}
                    />
                    <div className={styles.chartWrapper}>
                        <Text variant={'small'} className={styles.chartHeader}>
                            {accuracyChartHeader}
                        </Text>
                        <div className={styles.chartBody}>
                            <AccessibleChart
                                plotlyProps={accuracyPlot}
                                sharedSelectionContext={undefined}
                                theme={undefined}
                            />
                        </div>
                    </div>
                    <div className={styles.mainRight}>
                        <Text className={styles.rightTitle} block>
                            {localization.ModelComparison.howToRead}
                        </Text>
                        {howToReadAccuracySection}
                    </div>
                </div>
                <div className={styles.header}>
                    <Text variant={'mediumPlus'} className={styles.headerTitle}>
                        {localization.Report.outcomesTitle}
                    </Text>
                    <div className={styles.bannerWrapper}>
                        <div className={styles.headerBanner}>
                            <Text variant={'xxLargePlus'} className={styles.metricText} block>
                                {globalOutcomeString}
                            </Text>
                            <Text className={styles.firstMetricLabel} block>
                                {localization.formatString(
                                    localization.Report.globalAccuracyText,
                                    outcomeMetric.title.toLowerCase(),
                                )}
                            </Text>
                            <Text variant={'xxLargePlus'} className={styles.metricText} block>
                                {disparityOutcomeString}
                            </Text>
                            <Text className={styles.metricLabel} block>
                                {localization.formatString(
                                    localization.Report.accuracyDisparityText,
                                    outcomeMetric.title.toLowerCase(),
                                )}
                            </Text>
                        </div>
                    </div>
                </div>
                <div className={styles.presentationArea} style={{ height: `${areaHeights}px` }}>
                    <SummaryTable
                        binGroup={
                            this.props.dashboardContext.modelMetadata.featureNames[
                                this.props.featureBinPickerProps.selectedBinIndex
                            ]
                        }
                        binLabels={this.props.dashboardContext.groupNames}
                        formattedBinValues={formattedBinOutcomeValues}
                        metricLabel={outcomeMetric.title}
                        binValues={this.state.metrics.binnedOutcome}
                    />
                    <div className={styles.chartWrapper}>
                        <Text variant={'small'} className={styles.chartHeader} block>
                            {opportunityChartHeader}
                        </Text>
                        <div className={styles.chartBody}>
                            <AccessibleChart
                                plotlyProps={opportunityPlot}
                                sharedSelectionContext={undefined}
                                theme={undefined}
                            />
                        </div>
                    </div>
                    <div className={styles.mainRight}>
                        <Text className={styles.rightTitle} block>
                            {localization.ModelComparison.howToRead}
                        </Text>
                        <Text className={styles.rightText} block>
                            {howToReadOutcomesSection}
                        </Text>
                    </div>
                </div>
            </div>
        );
    }

    private readonly formatNumbers = (value: number, key: string, isRatio = false, sigDigits = 3): string => {
        if (value === null || value === undefined || value === NaN) {
            return NaN.toString();
        }
        const styleObject = { maximumSignificantDigits: sigDigits };
        if (AccuracyOptions[key] && AccuracyOptions[key].isPercentage && !isRatio) {
            (styleObject as any).style = 'percent';
        }
        return value.toLocaleString(undefined, styleObject);
    };

    private readonly clearModelSelection = (): void => {
        this.props.selections.onSelect([]);
    };

    private readonly onEditConfigs = (): void => {
        if (this.props.modelCount > 1) {
            this.props.selections.onSelect([]);
        }
        this.props.onEditConfigs();
    };

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
            const accuracy = await this.props.metricsCache.getMetric(
                this.props.dashboardContext.binVector,
                this.props.featureBinPickerProps.selectedBinIndex,
                this.props.selectedModelIndex,
                this.props.accuracyPickerProps.selectedAccuracyKey,
            );
            const accuracyDisparity = await this.props.metricsCache.getDisparityMetric(
                this.props.dashboardContext.binVector,
                this.props.featureBinPickerProps.selectedBinIndex,
                this.props.selectedModelIndex,
                this.props.accuracyPickerProps.selectedAccuracyKey,
                ParityModes.difference,
            );
            if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification) {
                binnedUnderprediction = (
                    await this.props.metricsCache.getMetric(
                        this.props.dashboardContext.binVector,
                        this.props.featureBinPickerProps.selectedBinIndex,
                        this.props.selectedModelIndex,
                        'underprediction',
                    )
                ).bins;
                binnedOverprediction = (
                    await this.props.metricsCache.getMetric(
                        this.props.dashboardContext.binVector,
                        this.props.featureBinPickerProps.selectedBinIndex,
                        this.props.selectedModelIndex,
                        'overprediction',
                    )
                ).bins;
                outcomes = await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex,
                    this.props.selectedModelIndex,
                    'selection_rate',
                );
                outcomeDisparity = await this.props.metricsCache.getDisparityMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex,
                    this.props.selectedModelIndex,
                    'selection_rate',
                    ParityModes.difference,
                );
            }
            if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.probability) {
                predictions = this.props.dashboardContext.predictions[this.props.selectedModelIndex];
                binnedOverprediction = (
                    await this.props.metricsCache.getMetric(
                        this.props.dashboardContext.binVector,
                        this.props.featureBinPickerProps.selectedBinIndex,
                        this.props.selectedModelIndex,
                        'overprediction',
                    )
                ).bins;
                binnedUnderprediction = (
                    await this.props.metricsCache.getMetric(
                        this.props.dashboardContext.binVector,
                        this.props.featureBinPickerProps.selectedBinIndex,
                        this.props.selectedModelIndex,
                        'underprediction',
                    )
                ).bins;
                outcomes = await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex,
                    this.props.selectedModelIndex,
                    'average',
                );
                outcomeDisparity = await this.props.metricsCache.getDisparityMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex,
                    this.props.selectedModelIndex,
                    'average',
                    ParityModes.difference,
                );
            }
            if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.regression) {
                predictions = this.props.dashboardContext.predictions[this.props.selectedModelIndex];
                errors = predictions.map((predicted, index) => {
                    return predicted - this.props.dashboardContext.trueY[index];
                });
                outcomes = await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex,
                    this.props.selectedModelIndex,
                    'average',
                );
                outcomeDisparity = await this.props.metricsCache.getDisparityMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex,
                    this.props.selectedModelIndex,
                    'average',
                    ParityModes.difference,
                );
            }
            this.setState({
                metrics: {
                    globalAccuracy: accuracy.global,
                    binnedAccuracy: accuracy.bins,
                    accuracyDisparity,
                    globalOutcome: outcomes.global,
                    binnedOutcome: outcomes.bins,
                    outcomeDisparity,
                    predictions,
                    errors,
                    binnedOverprediction,
                    binnedUnderprediction,
                },
            });
        } catch {
            // todo;
        }
    }
}
