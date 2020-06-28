import _ from 'lodash';
import { AccessibleChart, ChartBuilder, IPlotlyProperty, PlotlyMode, SelectionContext } from 'mlchartlib';
import { getTheme, Text } from 'office-ui-fabric-react';
import { ActionButton } from 'office-ui-fabric-react/lib/Button';
import { ChoiceGroup, IChoiceGroupOption } from 'office-ui-fabric-react/lib/ChoiceGroup';
import { Spinner, SpinnerSize } from 'office-ui-fabric-react/lib/Spinner';
import { Stack } from 'office-ui-fabric-react/lib/Stack';
import React from 'react';
import { AccuracyOptions } from '../AccuracyMetrics';
import { IAccuracyPickerProps, IFeatureBinPickerProps, IParityPickerProps } from '../FairnessWizard';
import { FormatMetrics } from '../FormatMetrics';
import { IFairnessContext } from '../IFairnessContext';
import { PredictionTypes } from '../IFairnessProps';
import { localization } from '../Localization/localization';
import { MetricsCache } from '../MetricsCache';
import { ParityModes } from '../ParityMetrics';
import { ModelComparisionChartStyles } from './ModelComparisionChart.styles';

const theme = getTheme();
export interface IModelComparisonProps {
    dashboardContext: IFairnessContext;
    selections: SelectionContext;
    metricsCache: MetricsCache;
    modelCount: number;
    accuracyPickerProps: IAccuracyPickerProps;
    parityPickerProps: IParityPickerProps;
    featureBinPickerProps: IFeatureBinPickerProps;
    onEditConfigs: () => void;
}

export interface IState {
    accuracyArray?: number[];
    disparityArray?: number[];
    disparityInOutcomes: boolean;
}

export class ModelComparisonChart extends React.PureComponent<IModelComparisonProps, IState> {
    private readonly plotlyProps: IPlotlyProperty = {
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
                datapointLevelAccessors: {
                    customdata: {
                        path: ['index'],
                        plotlyPath: 'customdata',
                    },
                },
                mode: PlotlyMode.markers,
                marker: {
                    size: 14,
                },
                type: 'scatter',
                xAccessor: 'Accuracy',
                yAccessor: 'Parity',
                hoverinfo: 'text',
            },
        ],
        layout: {
            autosize: true,
            font: {
                size: 10,
            },
            margin: {
                t: 4,
                r: 0,
            },
            hovermode: 'closest',
            xaxis: {
                automargin: true,
                fixedrange: true,
                mirror: true,
                linecolor: theme.semanticColors.disabledBorder,
                linewidth: 1,
                title: {
                    text: 'Error',
                },
            },
            yaxis: {
                automargin: true,
                fixedrange: true,
                title: {
                    text: 'Disparity',
                },
            },
        } as any,
    };

    constructor(props: IModelComparisonProps) {
        super(props);
        this.state = {
            disparityInOutcomes: true,
        };
    }

    public render(): React.ReactNode {
        const styles = ModelComparisionChartStyles();
        if (!this.state || this.state.accuracyArray === undefined || this.state.disparityArray === undefined) {
            this.loadData();
            return <Spinner className={styles.spinner} size={SpinnerSize.large} label={localization.calculating} />;
        }
        const data = this.state.accuracyArray.map((accuracy, index) => {
            return {
                Parity: this.state.disparityArray[index],
                Accuracy: accuracy,
                index: index,
            };
        });

        let minAccuracy: number = Number.MAX_SAFE_INTEGER;
        let maxAccuracy: number = Number.MIN_SAFE_INTEGER;
        let maxDisparity: number = Number.MIN_SAFE_INTEGER;
        let minDisparity: number = Number.MAX_SAFE_INTEGER;
        let minAccuracyIndex: number;
        let maxAccuracyIndex: number;
        let minDisparityIndex: number;
        let maxDisparityIndex: number;
        this.state.accuracyArray.forEach((value, index) => {
            if (value >= maxAccuracy) {
                maxAccuracyIndex = index;
                maxAccuracy = value;
            }
            if (value <= minAccuracy) {
                minAccuracyIndex = index;
                minAccuracy = value;
            }
        });
        this.state.disparityArray.forEach((value, index) => {
            if (value >= maxDisparity) {
                maxDisparityIndex = index;
                maxDisparity = value;
            }
            if (value <= minDisparity) {
                minDisparityIndex = index;
                minDisparity = value;
            }
        });
        const formattedMinAccuracy = FormatMetrics.formatNumbers(
            minAccuracy,
            this.props.accuracyPickerProps.selectedAccuracyKey,
        );
        const formattedMaxAccuracy = FormatMetrics.formatNumbers(
            maxAccuracy,
            this.props.accuracyPickerProps.selectedAccuracyKey,
        );
        const formattedMinDisparity = FormatMetrics.formatNumbers(
            minDisparity,
            this.props.accuracyPickerProps.selectedAccuracyKey,
        );
        const formattedMaxDisparity = FormatMetrics.formatNumbers(
            maxDisparity,
            this.props.accuracyPickerProps.selectedAccuracyKey,
        );
        let selectedMetric = AccuracyOptions[this.props.accuracyPickerProps.selectedAccuracyKey];
        // handle custom metric case
        if (selectedMetric === undefined) {
            selectedMetric = this.props.accuracyPickerProps.accuracyOptions.find(
                (metric) => metric.key === this.props.accuracyPickerProps.selectedAccuracyKey,
            );
        }
        const insights2 = localization.formatString(
            localization.ModelComparison.insightsText2,
            selectedMetric.title,
            formattedMinAccuracy,
            formattedMaxAccuracy,
            formattedMinDisparity,
            formattedMaxDisparity,
        );
        const metricTitleAppropriateCase = selectedMetric.alwaysUpperCase
            ? selectedMetric.title
            : selectedMetric.title.toLowerCase();
        const insights3 = localization.formatString(
            localization.ModelComparison.insightsText3,
            metricTitleAppropriateCase,
            selectedMetric.isMinimization ? formattedMinAccuracy : formattedMaxAccuracy,
            FormatMetrics.formatNumbers(
                this.state.disparityArray[selectedMetric.isMinimization ? minAccuracyIndex : maxAccuracyIndex],
                this.props.accuracyPickerProps.selectedAccuracyKey,
            ),
        );

        const insights4 = localization.formatString(
            localization.ModelComparison.insightsText4,
            metricTitleAppropriateCase,
            FormatMetrics.formatNumbers(
                this.state.accuracyArray[minDisparityIndex],
                this.props.accuracyPickerProps.selectedAccuracyKey,
            ),
            formattedMinDisparity,
        );

        const howToReadText = localization.formatString(
            localization.ModelComparison.howToReadText,
            this.props.modelCount.toString(),
            metricTitleAppropriateCase,
            selectedMetric.isMinimization ? localization.ModelComparison.lower : localization.ModelComparison.higher,
        );

        const props = _.cloneDeep(this.plotlyProps);
        props.data = ChartBuilder.buildPlotlySeries(props.data[0], data).map((series) => {
            series.name = this.props.dashboardContext.modelNames[series.name];
            series.text = this.props.dashboardContext.modelNames;
            return series;
        });
        const accuracyMetricTitle = selectedMetric.title;
        props.layout.xaxis.title = accuracyMetricTitle;
        props.layout.yaxis.title = this.state.disparityInOutcomes
            ? localization.ModelComparison.disparityInOutcomes
            : (localization.formatString(
                  localization.ModelComparison.disparityInAccuracy,
                  metricTitleAppropriateCase,
              ) as string);
        return (
            <Stack className={styles.frame}>
                <div className={styles.header}>
                    <Text variant={'large'} className={styles.headerTitle} block>
                        {localization.ModelComparison.title}
                    </Text>
                    <ActionButton
                        iconProps={{ iconName: 'Edit' }}
                        onClick={this.props.onEditConfigs}
                        className={styles.editButton}
                    >
                        {localization.Report.editConfiguration}
                    </ActionButton>
                </div>
                <div className={styles.main}>
                    <div className={styles.chart}>
                        <AccessibleChart
                            plotlyProps={props}
                            sharedSelectionContext={this.props.selections}
                            theme={undefined}
                        />
                    </div>
                    <div className={styles.mainRight}>
                        <Text className={styles.rightTitle} block>
                            {localization.ModelComparison.howToRead}
                        </Text>
                        <Text className={styles.rightText} block>
                            {howToReadText}
                        </Text>
                        <Text className={styles.insights} block>
                            {localization.ModelComparison.insights}
                        </Text>
                        <div className={styles.insightsText}>
                            <Text className={styles.textSection} block>
                                {insights2}
                            </Text>
                            <Text className={styles.textSection} block>
                                {insights3}
                            </Text>
                            <Text className={styles.textSection} block>
                                {insights4}
                            </Text>
                        </div>
                    </div>
                </div>
                <div>
                    <ChoiceGroup
                        className={styles.radio}
                        selectedKey={this.state.disparityInOutcomes ? 'outcomes' : 'accuracy'}
                        options={[
                            {
                                key: 'accuracy',
                                text: localization.formatString(
                                    localization.ModelComparison.disparityInAccuracy,
                                    metricTitleAppropriateCase,
                                ) as string,
                                styles: { choiceFieldWrapper: styles.radioOptions },
                            },
                            {
                                key: 'outcomes',
                                text: localization.ModelComparison.disparityInOutcomes,
                                styles: { choiceFieldWrapper: styles.radioOptions },
                            },
                        ]}
                        onChange={this.disparityChanged}
                        label={localization.ModelComparison.howToMeasureDisparity}
                        required={false}
                    ></ChoiceGroup>
                </div>
            </Stack>
        );
    }

    private async loadData(): Promise<void> {
        try {
            const accuracyPromises = new Array(this.props.modelCount).fill(0).map((unused, modelIndex) => {
                return this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex,
                    modelIndex,
                    this.props.accuracyPickerProps.selectedAccuracyKey,
                );
            });
            const disparityMetric = this.state.disparityInOutcomes
                ? this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification
                    ? 'selection_rate'
                    : 'average'
                : this.props.accuracyPickerProps.selectedAccuracyKey;
            const disparityPromises = new Array(this.props.modelCount).fill(0).map((unused, modelIndex) => {
                return this.props.metricsCache.getDisparityMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex,
                    modelIndex,
                    disparityMetric,
                    ParityModes.difference,
                );
            });

            const accuracyArray = (await Promise.all(accuracyPromises)).map((metric) => metric.global);
            const disparityArray = await Promise.all(disparityPromises);
            this.setState({ accuracyArray, disparityArray });
        } catch {
            // todo;
        }
    }

    private readonly disparityChanged = (ev: React.FormEvent<HTMLInputElement>, option: IChoiceGroupOption): void => {
        const disparityInOutcomes = option.key !== 'accuracy';
        if (this.state.disparityInOutcomes !== disparityInOutcomes) {
            this.setState({ disparityInOutcomes, disparityArray: undefined });
        }
    };
    // TODO: Reuse if multiselect re-enters design
    // private readonly applySelections = (chartId: string, selectionIds: string[], plotlyProps: IPlotlyProperty) => {
    //     if (!plotlyProps.data || plotlyProps.data.length === 0) {
    //         return;
    //     }
    //     const customData: string[] = (plotlyProps.data[0] as any).customdata;
    //     if (!customData) {
    //         return;
    //     }
    //     const colors = customData.map(modelIndex => {
    //         const selectedIndex = this.props.selections.selectedIds.indexOf(modelIndex);
    //         if (selectedIndex !== -1) {
    //             return FabricStyles.plotlyColorPalette[selectedIndex % FabricStyles.plotlyColorPalette.length];
    //         }
    //         return "#111111";
    //     });
    //     const shapes = customData.map(modelIndex => {
    //         const selectedIndex = this.props.selections.selectedIds.indexOf(modelIndex);
    //         if (selectedIndex !== -1) {
    //             return 1
    //         }
    //         return 0;
    //     });
    //     Plotly.restyle(chartId, 'marker.color' as any, [colors] as any);
    //     Plotly.restyle(chartId, 'marker.symbol' as any, [shapes] as any);
    // }
}
