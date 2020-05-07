import React from "react";
import { localization } from "../Localization/localization";
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import { DataSpecificationBlade } from "./DataSpecificationBlade";
import { Separator } from "office-ui-fabric-react/lib/Separator";
import { IWizardTabProps } from "../IWizardTabProps";
import { WizardFooter } from "./WizardFooter";
import { TileList, ITileProp } from "./TileList";
import { IParityPickerProps } from "../FairnessWizard";
import { mergeStyleSets } from "@uifabric/styling";

export interface IParityTabProps extends IWizardTabProps {
    parityPickerProps: IParityPickerProps;
}

export class ParityTab extends React.PureComponent<IParityTabProps> {
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
            width: "750px",
            height: "100%",
        },
        main: {
            height: "100%",
            minWidth: "550px",
            flex: 1
        },
        header: {
            color: "#333333",
            fontSize: "32px",
            lineHeight: "40px",
            fontWeight: "300",
            margin: "26px 0"
        },
        textBody: {
            color: "#333333",
            paddingTop: "12px",
            fontSize: "15px",
            lineHeight: "20px",
            fontWeight: "normal",
            paddingBottom: "12px"
        }
    });
    render(): React.ReactNode {
        return(
            <Stack horizontal horizontalAlign="space-between" className={ParityTab.classNames.frame}>
                <StackItem grow={2}>
                    <Stack className={ParityTab.classNames.main}>
                        <h2 className={ParityTab.classNames.header}>
                            {localization.Parity.header}
                        </h2>
                        <p className={ParityTab.classNames.textBody}>{localization.Parity.body}</p>
                        <StackItem grow={2} className={ParityTab.classNames.itemsList}>
                            <TileList
                                items={this.props.parityPickerProps.parityOptions.map((parity, index): ITileProp => {
                                    const selected = this.props.parityPickerProps.selectedParityKey === parity.key;
                                    return {
                                        title: parity.title,
                                        description: parity.description,
                                        onSelect: this.props.parityPickerProps.onParityChange.bind(this, parity.key),
                                        selected
                                    };
                                })}
                            />
                        </StackItem>
                        <Separator />
                        <WizardFooter onNext={this.props.onNext} onPrevious={this.props.onPrevious}/>
                    </Stack>
                </StackItem>
                <DataSpecificationBlade
                        numberRows={this.props.dashboardContext.trueY.length}
                        featureNames={this.props.dashboardContext.modelMetadata.featureNames}/>
            </Stack>
        );
    }
}