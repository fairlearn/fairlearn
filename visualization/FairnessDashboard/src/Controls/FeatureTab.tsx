import React from "react";
import { localization } from "../Localization/localization";
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import { DataSpecificationBlade } from "./DataSpecificationBlade";
import { Separator } from "office-ui-fabric-react/lib/Separator";
import { IWizardTabProps } from "../IWizardTabProps";
import { WizardFooter } from "./WizardFooter";
import { List } from "office-ui-fabric-react/lib/List";
import { mergeStyleSets } from "@uifabric/styling";
import { Icon } from "office-ui-fabric-react/lib/Icon";
import { IBinnedResponse } from "../IBinnedResponse";
import { Text } from "office-ui-fabric-react/lib/Text";
import { Modal } from 'office-ui-fabric-react/lib/Modal';
import { ActionButton } from "office-ui-fabric-react/lib/Button";
import BinDialog from "./BinDialog";
import { INumericRange, RangeTypes } from "mlchartlib";

interface IFeatureItem {
    title: string;
    description: string;
    onSelect: (index: number) => void;
    selected: boolean;
    categories?: string[];
}

export interface IFeatureTabProps extends IWizardTabProps {
    featureBins: IBinnedResponse[];
    selectedFeatureIndex: number;
    selectedFeatureChange: (value: number) => void;
    saveBin: (bin: IBinnedResponse) => void;
}

interface IState {
    expandedBins: number[];
    editingFeatureIndex: number | undefined;
}

export class FeatureTab extends React.PureComponent<IFeatureTabProps, IState> {
    private static readonly classNames = mergeStyleSets({
        itemCell: [
          {
            padding: "30px 36px 20px 0",
            width: "100%",
            position: "relative",
            float: "left",
            cursor: "pointer",
            boxSizing: "border-box",
            borderBottom: "1px solid #CCCCCC",
            selectors: {
              '&:hover': { background: "lightgray" }
            }
          }
        ],
        iconClass: {
            fontSize: "20px",
            position: "absolute",
            right: "10px",
            top: "10px"
        },
        itemsList: {
            overflowY: "auto"
        },
        frame: {
            height: "100%",
        },
        main: {
            height: "100%",
            maxWidth: "700px",
            flex: 1
        },
        header: {
            color: "#333333",
            fontSize: "32px",
            lineHeight: "39px",
            fontWeight: "100"
        },
        textBody: {
            paddingTop: "12px",
            fontSize: "18px",
            lineHeight: "24px",
            fontWeight: "300"
        },
        tableHeader: {
            borderBottom: "1px solid #CCCCCC"
        },
        itemTitle: {
            margin: 0,
            color: "#333333",
            fontSize: "22px",
            lineHeight: "26px",
            fontWeight: "300"
        },
        valueCount: {
            paddingTop: "15px",
            color: "#333333",
            fontSize: "15px",
            lineHeight: "18px",
            fontWeight: "500"
        }
    });

    constructor(props: IFeatureTabProps) {
        super(props);
        this.state = { 
            expandedBins: [],
            editingFeatureIndex: undefined
        };
    }
    
    render(): React.ReactNode {
        return(
            <Stack horizontal horizontalAlign="space-between" className={FeatureTab.classNames.frame}>
                <Modal
                    isOpen={this.state.editingFeatureIndex !== undefined}
                    isBlocking={false}
                    onDismiss={this.hideModal}>
                    { this.state.editingFeatureIndex !== undefined && <BinDialog
                        range={this.props.dashboardContext.modelMetadata.featureRanges[this.state.editingFeatureIndex] as INumericRange}
                        bins={this.props.featureBins[this.state.editingFeatureIndex]}
                        dataset={this.props.dashboardContext.dataset}
                        index={this.state.editingFeatureIndex}
                        onSave={this.onBinSave}
                        onCancel={this.hideModal}/>}
                </Modal>
                <Stack className={FeatureTab.classNames.main}>
                    <h2 className={FeatureTab.classNames.header}>
                        {localization.Feature.header}
                    </h2>
                    <p className={FeatureTab.classNames.textBody}>{localization.Feature.body}</p>
                    <div className={FeatureTab.classNames.tableHeader}></div>
                    <StackItem grow={2} className={FeatureTab.classNames.itemsList}>
                        <List
                            items={this.props.featureBins}
                            onRenderCell={this._onRenderCell}
                        />
                    </StackItem>
                    <WizardFooter onNext={this.props.onNext}/>
                </Stack>
                <DataSpecificationBlade
                        numberRows={this.props.dashboardContext.dataset.length}
                        featureNames={this.props.dashboardContext.modelMetadata.featureNames}/>
            </Stack>
        );
    }

    private readonly hideModal = (): void => {
        this.setState({editingFeatureIndex: undefined});
    }

    private readonly onBinSave = (bin: IBinnedResponse): void => {
        this.setState({editingFeatureIndex: undefined});
        this.props.saveBin(bin);
    }

    private readonly editBins = (index: number) => {
        this.setState({editingFeatureIndex: index});
    }

    private readonly _onRenderCell = (item: IBinnedResponse, index: number | undefined): JSX.Element => {
        return (
          <div
            key={index}
            className={FeatureTab.classNames.itemCell}
            onClick={this.props.selectedFeatureChange.bind(this, index)}
            data-is-focusable={true}
          >
            {this.props.selectedFeatureIndex === index && (<Icon iconName="CompletedSolid" className={FeatureTab.classNames.iconClass}/>)}
            <h2 className={FeatureTab.classNames.itemTitle}>{this.props.dashboardContext.modelMetadata.featureNames[index]}</h2>
            {item.rangeType === RangeTypes.categorical &&
                <p className={FeatureTab.classNames.valueCount}>{localization.formatString(localization.Feature.summaryCategoricalCount, item.array.length) as string}</p>
            }
            {!this.props.dashboardContext.modelMetadata.featureIsCategorical[index] && 
                <ActionButton 
                    onClick={this.editBins.bind(this, index)}>{localization.Feature.editBinning}</ActionButton>
            }
            {!this.state.expandedBins.includes(index) && (
                <ActionButton
                    iconProps={{iconName: "Forward"}}
                    onClick={this.updateExpandedList.bind(this, index)}>{localization.Feature.showCategories}</ActionButton>)}
            {this.state.expandedBins.includes(index) && (
                <div>
                    <ActionButton 
                    iconProps={{iconName: "Back"}}
                    onClick={this.updateExpandedList.bind(this)}>{localization.Feature.hideCategories}</ActionButton>
                    {!!item.labelArray && item.labelArray.map((category, index) => <div key={index}>{category}</div>)}
                </div>)}
          </div>
        );
    }

    private readonly updateExpandedList = (value?: number): void => {
        this.setState(() => {return {expandedBins: [value]}})
    }
}