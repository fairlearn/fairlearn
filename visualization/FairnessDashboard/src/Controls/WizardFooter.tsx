import React from "react";
import { DefaultButton, PrimaryButton } from "office-ui-fabric-react/lib/Button";
import { localization } from "../Localization/localization";
import { mergeStyleSets } from "@uifabric/styling";

export interface IWizardFooterProps {
    onNext: () => void;
    onPrevious?: () => void;
}

export class WizardFooter extends React.PureComponent<IWizardFooterProps> {
    private static readonly classNames = mergeStyleSets({
        frame: {
            display: "inline-flex",
            flexDirection: "row-reverse",
            paddingTop: "10px",
            paddingBottom: "10px"
        },
        next: {
            height: "40px",
            padding: "12px",
            color: "#FFFFFF",
            fontSize: "12px",
            lineHeight: "16px",
            background: "linear-gradient(338.45deg, #5A53FF -73.33%, #5A53FF 84.28%)",
            fontWeight: "600",
            fontStyle: "normal",
            marginLeft: "10px",
            borderRadius: "5px"
        },
        back: {
            height: "40px",
            padding: "12px",
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            backgroundColor: "#FFFFFF",
            fontWeight: "600",
            fontStyle: "normal",
            borderRadius: "5px"
        }
    });
    
    public render(): React.ReactNode {
        return (
            <div className={WizardFooter.classNames.frame}>
                <PrimaryButton className={WizardFooter.classNames.next} text={localization.Footer.next} onClick={this.props.onNext}/>
                {(!!this.props.onPrevious) &&
                    <DefaultButton
                        className={WizardFooter.classNames.back}
                        text={localization.Footer.back}
                        onClick={this.props.onPrevious}/>
                }
            </div>
        );
    }
}