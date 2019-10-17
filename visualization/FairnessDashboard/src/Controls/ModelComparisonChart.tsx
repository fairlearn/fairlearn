import React from "react";
import { AccessibleChart, ChartBuilder, IPlotlyProperty, PlotlyMode, SelectionContext } from "mlchartlib";
import { IFairnessContext } from "../IFairnessContext";
import _ from "lodash";
import { MetricsCache } from "../MetricsCache";
import { IAccuracyPickerProps, IParityPickerProps, IFeatureBinPickerProps } from "../FairnessWizard";
import { ParityModes } from "../ParityMetrics";
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import { Text } from "office-ui-fabric-react/lib/Text";
import { localization } from "../Localization/localization";
import { Spinner, SpinnerSize } from "office-ui-fabric-react/lib/Spinner";
import { mergeStyleSets } from "@uifabric/styling";
import { ActionButton } from "office-ui-fabric-react/lib/Button";

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
}

export class ModelComparisonChart extends React.PureComponent<IModelComparisonProps, IState> {
    private readonly plotlyProps: IPlotlyProperty = {
        config: { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian', 'lasso2d', 'select2d'] },
        data: [
            {
                datapointLevelAccessors: {
                    customdata: {
                        path: ['index'],
                        plotlyPath: 'customdata'
                    }
                },
                mode: PlotlyMode.markers,
                marker: {
                    size: 14
                },
                type: 'scatter',
                xAccessor: 'Accuracy',
                yAccessor: 'Parity'
            }
        ],
        layout: {
            autosize: true,
            font: {
                size: 10
            },
            margin: {
                t: 10
            },
            hovermode: 'closest',
            xaxis: {
                automargin: true,
                title:{
                    text: 'Error'
                }
            },
            yaxis: {
                automargin: true,
                title:{
                    text: 'Disparity'
                }
            },
        } as any
    };

    private static readonly classNames = mergeStyleSets({
        spinner: {
            margin: "auto",
            fontFamily: `"Segoe UI", "Segoe UI Web (West European)", "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", sans-serif`,
            padding: "40px"
        }
    });

    public render(): React.ReactNode {
        if (!this.state || this.state.accuracyArray === undefined || this.state.disparityArray === undefined) {
            this.loadData();
            return (
                <Spinner className={ModelComparisonChart.classNames.spinner} size={SpinnerSize.large} label={localization.calculating}/>
            );
        }
        const data = this.state.accuracyArray.map((accuracy, index) => {

            return {
                Parity: this.state.disparityArray[index],
                Accuracy: accuracy,
                index: index
            };
        });
        
        const props = _.cloneDeep(this.plotlyProps);
        props.data = ChartBuilder.buildPlotlySeries(props.data[0], data).map(series => {
            series.name = this.props.dashboardContext.modelNames[series.name];
            return series;
        });
        return (
            <Stack styles={{root: {height: "100%"}}}>
                <StackItem styles={{root: {backgroundColor: "#EBEBEB", padding: "10px 30px", height: "150px"}}}>
                    <Stack horizontal horizontalAlign="space-between" verticalAlign="center" styles={{root: {height: "100%"}}}>
                        <Text variant={"xxLarge"}>{localization.Report.title}</Text>
                        <ActionButton iconProps={{iconName: "Edit"}} onClick={this.props.onEditConfigs}>{localization.Report.editConfiguration}</ActionButton>
                    </Stack>
                </StackItem>
                <StackItem grow={2}>
                    <Stack horizontal horizontalAlign="space-between" styles={{root: {height: "100%"}}}>
                        <StackItem grow={2}>
                            <AccessibleChart
                                plotlyProps={props}
                                sharedSelectionContext={this.props.selections}
                                theme={undefined}
                            />
                        </StackItem>
                        <p style={{width: "300px"}}>"Lorem Ipsum goes here......."</p>
                    </Stack>
                </StackItem>
            </Stack>);
    }

    private async loadData(): Promise<void> {
        try {
            const accuracyPromises = new Array(this.props.modelCount).fill(0).map((unused, modelIndex) => {
                return this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex,
                    modelIndex,
                    this.props.accuracyPickerProps.selectedAccuracyKey);
            });
            const disparityPromises = new Array(this.props.modelCount).fill(0).map((unused, modelIndex) => {
                return this.props.metricsCache.getDisparityMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex,
                    modelIndex, this.props.parityPickerProps.selectedParityKey,
                    ParityModes.difference);
            });

            const accuracyArray = (await Promise.all(accuracyPromises)).map(metric => metric.global);
            const disparityArray = await Promise.all(disparityPromises);
            this.setState({accuracyArray, disparityArray});
        } catch {
            // todo;
        }
    }
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