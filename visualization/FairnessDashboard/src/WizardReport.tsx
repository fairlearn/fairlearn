import { getTheme } from '@uifabric/styling';
import _ from 'lodash';
import { AccessibleChart, IPlotlyProperty } from 'mlchartlib';
import { ActionButton, IconButton, PrimaryButton } from 'office-ui-fabric-react/lib/Button';
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
import { IDropdownStyles, IDropdownOption, Dropdown, Modal, IIconProps, Icon } from 'office-ui-fabric-react';
import { OverallTable } from './Controls/OverallTable';

const theme = getTheme();
interface IMetrics {
    globalAccuracy: number;
    binnedAccuracy: number[];
    accuracyDisparity: number;
    globalOutcome: number;
    outcomeDisparity: number;
    binnedOutcome: number[];
    // Optional, based on model type
    globalOverprediction?: number;
    globalUnderprediction?: number;
    binnedOverprediction?: number[];
    binnedUnderprediction?: number[];
    // different length, raw unbinned errors and predictions
    errors?: number[];
    predictions?: number[];
}

export interface IState {
    metrics?: IMetrics;
    featureKey?: string;
    showModalHelp?: boolean;
    expandAttributes: boolean;
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
            } as any,
        ],
        layout: {
            autosize: true,
            barmode: 'relative',
            colorway: ChartColors,
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
        const dropdownStyles: Partial<IDropdownStyles> = {
            dropdown: { width: 180 },
            title: { borderWidth: '1px', borderRadius: '5px' },
        };

        const iconButtonStyles = {
            root: {
                color: theme.semanticColors.bodyText,
                marginLeft: 'auto',
                marginTop: '4px',
                marginRight: '2px',
            },
            rootHovered: {
                color: theme.semanticColors.bodyBackgroundHovered,
            },
        };

        const featureOptions: IDropdownOption[] = this.props.dashboardContext.modelMetadata.featureNames.map((x) => {
            return { key: x, text: x };
        });

        const alternateHeight =
            this.props.featureBinPickerProps.featureBins[this.props.featureBinPickerProps.selectedBinIndex].labelArray
                .length *
                60 +
            106;
        const areaHeights = Math.max(300, alternateHeight);

        const accuracyKey = this.props.accuracyPickerProps.selectedAccuracyKey;
        const outcomeKey =
            this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification
                ? 'selection_rate'
                : 'average';
        const outcomeMetric = AccuracyOptions[outcomeKey];

        const overpredicitonKey = 'overprediction';
        const underpredictionKey = 'underprediction';

        const accuracyPlot = _.cloneDeep(WizardReport.barPlotlyProps);
        const opportunityPlot = _.cloneDeep(WizardReport.barPlotlyProps);
        const nameIndex = this.props.dashboardContext.groupNames.map((unuxed, i) => i);
        let howToReadAccuracySection: React.ReactNode;
        let insightsAccuracySection: React.ReactNode;
        let howToReadOutcomesSection: React.ReactNode;
        let insightsOutcomesSection: React.ReactNode;
        let accuracyChartHeader = '';
        let opportunityChartHeader = '';

        let mainChart;
        if (!this.state || !this.state.metrics) {
            this.loadData();
            mainChart = (
                <Spinner className={styles.spinner} size={SpinnerSize.large} label={localization.calculating} />
            );
        } else {
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
            const formattedBinOverPredictionValues = this.state.metrics.binnedOverprediction.map((value) =>
                this.formatNumbers(value, overpredicitonKey),
            );
            const formattedBinUnderPredictionValues = this.state.metrics.binnedUnderprediction.map((value) =>
                this.formatNumbers(value, underpredictionKey),
            );

            const globalOverpredictionString = this.formatNumbers(this.state.metrics.globalOverprediction, outcomeKey);
            const globalUnderpredictionString = this.formatNumbers(
                this.state.metrics.globalUnderprediction,
                outcomeKey,
            );

            const overallMetrics = [
                globalAccuracyString,
                globalOutcomeString,
                globalOverpredictionString,
                globalUnderpredictionString,
            ];
            const formattedBinValues = [
                formattedBinAccuracyValues,
                formattedBinOutcomeValues,
                formattedBinOverPredictionValues,
                formattedBinUnderPredictionValues,
            ];
            const metricLabels = [
                AccuracyOptions[accuracyKey].title,
                AccuracyOptions[outcomeKey].title,
                AccuracyOptions[overpredicitonKey].title,
                AccuracyOptions[underpredictionKey].title,
            ];

            const InsightsIcon = () => <Icon iconName="CRMCustomerInsightsApp" className={styles.insightsIcon} />;
            const DownloadIcon = () => <Icon iconName="Download" className={styles.downloadIcon} />;

            const ChevronUp = () => <Icon iconName="ChevronUp" className={styles.chevronIcon} />;
            const ChevronDown = () => <Icon iconName="ChevronDown" className={styles.chevronIcon} />;

            const cancelIcon: IIconProps = { iconName: 'Cancel' };

            mainChart = (
                <div className={styles.main}>
                    <div className={styles.mainLeft}>
                        <div
                            className={styles.overallArea}
                            style={{
                                height:
                                    (!this.state.expandAttributes && '150px') ||
                                    (this.state.expandAttributes && `${150 + 50 * (areaHeights / 150)}px`),
                            }}
                        >
                            <OverallTable
                                binGroup={
                                    this.props.dashboardContext.modelMetadata.featureNames[
                                        this.props.featureBinPickerProps.selectedBinIndex
                                    ]
                                }
                                binLabels={this.props.dashboardContext.groupNames}
                                formattedBinValues={formattedBinValues}
                                metricLabels={metricLabels}
                                overallMetrics={overallMetrics}
                                expandAttributes={this.state.expandAttributes}
                                binValues={this.state.metrics.binnedAccuracy}
                            />
                        </div>
                        <div className={styles.expandAttributes} onClick={this.expandAttributes}>
                            {(this.state.expandAttributes && <ChevronUp />) ||
                                (!this.state.expandAttributes && <ChevronDown />)}
                            <Text>
                                {(this.state.expandAttributes && localization.Report.collapseSensitiveAttributes) ||
                                    (!this.state.expandAttributes && localization.Report.expandSensitiveAttributes)}
                            </Text>
                        </div>
                        <div className={styles.equalizedOdds}>
                            <Text>{localization.Report.equalizedOddsDisparity}</Text>
                        </div>
                        <div className={styles.howTo}>
                            <ActionButton onClick={this.handleOpenModalHelp}>
                                <div className={styles.infoButton}>i</div>
                                {localization.ModelComparison.howToRead}
                            </ActionButton>
                            <Modal
                                titleAriaId="intro modal"
                                isOpen={this.state.showModalHelp}
                                onDismiss={this.handleCloseModalHelp}
                                isModeless={true}
                                containerClassName={styles.modalContentHelp}
                            >
                                <div style={{ display: 'flex' }}>
                                    <IconButton
                                        styles={iconButtonStyles}
                                        iconProps={cancelIcon}
                                        ariaLabel="Close popup modal"
                                        onClick={this.handleCloseModalHelp}
                                    />
                                </div>
                                <p className={styles.modalContentHelpText}>
                                    {localization.Report.classificationAccuracyHowToRead1}
                                    <br />
                                    <br />
                                    {localization.Report.classificationAccuracyHowToRead2}
                                    <br />
                                    <br />
                                    {localization.Report.classificationAccuracyHowToRead3}
                                    <br />
                                    <br />
                                </p>
                                <div style={{ display: 'flex', paddingBottom: '20px' }}>
                                    <PrimaryButton className={styles.doneButton} onClick={this.handleCloseModalHelp}>
                                        {localization.done}
                                    </PrimaryButton>
                                </div>
                            </Modal>
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
                                <div className={styles.chartHeader}>{accuracyChartHeader}</div>
                                <div className={styles.chartBody}>
                                    <AccessibleChart
                                        plotlyProps={accuracyPlot}
                                        sharedSelectionContext={undefined}
                                        theme={undefined}
                                    />
                                </div>
                            </div>
                        </div>
                        <div className={styles.legendPanel}>
                            <div className={styles.textRow}>
                                <div className={styles.colorBlock} style={{ backgroundColor: ChartColors[1] }} />
                                <div>
                                    <div className={styles.legendTitle}>{localization.Report.underestimationError}</div>
                                    <div className={styles.legendSubtitle}>
                                        {localization.Report.underpredictionExplanation}
                                    </div>
                                </div>
                            </div>
                            <div className={styles.textRow}>
                                <div className={styles.colorBlock} style={{ backgroundColor: ChartColors[0] }} />
                                <div>
                                    <div className={styles.legendTitle}>{localization.Report.overestimationError}</div>
                                    <div className={styles.legendSubtitle}>
                                        {localization.Report.overpredictionExplanation}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div className={styles.mainRight}>
                        <div className={styles.insights}>
                            <InsightsIcon />
                            <Text style={{ verticalAlign: 'middle' }}>{localization.ModelComparison.insights}</Text>
                        </div>
                        <div className={styles.insightsText}>{localization.loremIpsum}</div>
                        <div className={styles.downloadReport}>
                            <DownloadIcon />
                            <Text style={{ verticalAlign: 'middle' }}>
                                {localization.ModelComparison.downloadReport}
                            </Text>
                        </div>
                    </div>
                </div>
            );
        }

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
                        </div>
                    )}
                    <div className={styles.modelLabel}>
                        {localization.Report.assessmentResults}{' '}
                        <b>{this.props.dashboardContext.modelNames[this.props.selectedModelIndex]}</b>
                    </div>
                    <div className={styles.headerOptions}>
                        <Dropdown
                            className={styles.dropDown}
                            // label="Feature"
                            defaultSelectedKey={
                                this.props.dashboardContext.modelMetadata.featureNames[
                                    this.props.featureBinPickerProps.selectedBinIndex
                                ]
                            }
                            options={featureOptions}
                            disabled={false}
                            onChange={this.featureChanged}
                            styles={dropdownStyles}
                        />
                    </div>
                </div>
                {mainChart}
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

    private readonly expandAttributes = (): void => {
        this.setState({ expandAttributes: !this.state.expandAttributes });
    };

    private readonly onEditConfigs = (): void => {
        if (this.props.modelCount > 1) {
            this.props.selections.onSelect([]);
        }
        this.props.onEditConfigs();
    };

    private readonly handleOpenModalHelp = (event): void => {
        this.setState({ showModalHelp: true });
    };

    private readonly handleCloseModalHelp = (event): void => {
        this.setState({ showModalHelp: false });
    };

    private readonly featureChanged = (ev: React.FormEvent<HTMLInputElement>, option: IDropdownOption): void => {
        const featureKey = option.key.toString();
        if (this.state.featureKey !== featureKey) {
            this.props.featureBinPickerProps.selectedBinIndex = this.props.dashboardContext.modelMetadata.featureNames.indexOf(
                featureKey,
            );
            this.setState({ featureKey: featureKey, metrics: undefined });
        }
    };

    private async loadData(): Promise<void> {
        try {
            let binnedFNR: number[];
            let binnedFPR: number[];
            let overallOverprediction: number;
            let overallUnderprediction: number;
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
                overallUnderprediction = (
                    await this.props.metricsCache.getMetric(
                        this.props.dashboardContext.binVector,
                        this.props.featureBinPickerProps.selectedBinIndex,
                        this.props.selectedModelIndex,
                        'underprediction',
                    )
                ).global;
                binnedOverprediction = (
                    await this.props.metricsCache.getMetric(
                        this.props.dashboardContext.binVector,
                        this.props.featureBinPickerProps.selectedBinIndex,
                        this.props.selectedModelIndex,
                        'overprediction',
                    )
                ).bins;
                overallOverprediction = (
                    await this.props.metricsCache.getMetric(
                        this.props.dashboardContext.binVector,
                        this.props.featureBinPickerProps.selectedBinIndex,
                        this.props.selectedModelIndex,
                        'overprediction',
                    )
                ).global;
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
                    accuracyDisparity: accuracyDisparity,
                    globalOutcome: outcomes.global,
                    binnedOutcome: outcomes.bins,
                    outcomeDisparity: outcomeDisparity,
                    predictions: predictions,
                    errors: errors,
                    globalOverprediction: overallOverprediction,
                    globalUnderprediction: overallUnderprediction,
                    binnedOverprediction: binnedOverprediction,
                    binnedUnderprediction: binnedUnderprediction,
                },
            });
        } catch {
            // todo;
        }
    }
}
