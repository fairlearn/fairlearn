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
            height: "44px",
            padding: "12px",
            color: "#FFFFFF",
            fontSize: "18px",
            lineHeight: "24px",
            backgroundColor: "#666666",
            fontWeight: "400",
            marginLeft: "10px"
        },
        back: {
            height: "44px",
            padding: "12px",
            color: "#333333",
            fontSize: "18px",
            lineHeight: "24px",
            backgroundColor: "#FFFFFF",
            fontWeight: "400"
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