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
import { ActionButton } from "office-ui-fabric-react/lib/Button";

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
}

interface IState {
    expandedBins: number[];
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
        this.state = { expandedBins: []};
    }
    
    render(): React.ReactNode {
        return(
            <Stack horizontal horizontalAlign="space-between" className={FeatureTab.classNames.frame}>
                <Stack className={FeatureTab.classNames.main}>
                    <h2 className={FeatureTab.classNames.header}>
                        {localization.Feature.header}
                    </h2>
                    <p className={FeatureTab.classNames.textBody}>{localization.Feature.body}</p>
                    <div className={FeatureTab.classNames.tableHeader}></div>
                    <StackItem grow={2} className={FeatureTab.classNames.itemsList}>
                        <List
                            items={this.props.featureBins.map((bin, index) => {
                                return {
                                    title: this.props.dashboardContext.modelMetadata.featureNames[bin.featureIndex],
                                    description: localization.formatString(localization.Feature.summaryCategoricalCount, bin.array.length) as string,
                                    onSelect: this.props.selectedFeatureChange.bind(this, bin.featureIndex),
                                    selected: this.props.selectedFeatureIndex === bin.featureIndex,
                                    categories: bin.array as string[]
                                };
                            })}
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

    private readonly _onRenderCell = (item: IFeatureItem, index: number | undefined): JSX.Element => {
        return (
          <div
            key={index}
            className={FeatureTab.classNames.itemCell}
            onClick={item.onSelect.bind(this)}
            data-is-focusable={true}
          >
            {item.selected && (<Icon iconName="CompletedSolid" className={FeatureTab.classNames.iconClass}/>)}
            <h2 className={FeatureTab.classNames.itemTitle}>{item.title}</h2>
            <p className={FeatureTab.classNames.valueCount}>{item.description}</p>
            {!this.state.expandedBins.includes(index) && (
                <ActionButton 
                    iconProps={{iconName: "Forward"}}
                    onClick={this.updateExpandedList.bind(this, index)}>{localization.Feature.showCategories}</ActionButton>)}
            {this.state.expandedBins.includes(index) && (
                <div>
                    <ActionButton 
                    iconProps={{iconName: "Back"}}
                    onClick={this.updateExpandedList.bind(this)}>{localization.Feature.hideCategories}</ActionButton>
                    {!!item.categories && item.categories.map((category, index) => <div key={index}>{category}</div>)}
                </div>)}
          </div>
        );
    }

    private readonly updateExpandedList = (value?: number): void => {
        this.setState(() => {return {expandedBins: [value]}})
    }
}