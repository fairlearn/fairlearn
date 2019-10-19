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

export class FeatureTab extends React.PureComponent<IFeatureTabProps> {
    private static readonly classNames = mergeStyleSets({
        itemCell: [
          {
            padding: 10,
            width: "100%",
            height: "150px",
            position: "relative",
            float: "left",
            cursor: "pointer",
            boxSizing: "border-box",
            border: `1px solid grey`,
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
        }
    });
    
    render(): React.ReactNode {
        return(
            <Stack horizontal style={{height: "100%"}}>
                <StackItem grow={2}>
                    <Stack style={{height: "100%"}}>
                        <h2 style={{fontWeight: "bold"}}>
                            {localization.Feature.header}
                        </h2>
                        <p>{localization.Feature.body}</p>
                        <a>{localization.Feature.learnMore}</a>
                        <StackItem grow={2} className={FeatureTab.classNames.itemsList}>
                            <List
                                items={this.props.featureBins.map((bin, index) => {
                                    return {
                                        title: this.props.dashboardContext.modelMetadata.featureNames[bin.featureIndex],
                                        description: localization.formatString(localization.Feature.summaryCategoricalCount, bin.array.length),
                                        onSelect: this.props.selectedFeatureChange.bind(this, bin.featureIndex),
                                        selected: this.props.selectedFeatureIndex === bin.featureIndex
                                    };
                                })}
                                onRenderCell={this._onRenderCell}
                            />
                        </StackItem>
                        <Separator />
                        <WizardFooter onNext={this.props.onNext}/>
                    </Stack>
                </StackItem>
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
            <h2>{item.title}</h2>
            <p>{item.description}</p>
            <Text variant={"medium"}>"See categories"</Text>
          </div>
        );
    }
}