import { DefaultButton, PrimaryButton } from "office-ui-fabric-react/lib/Button";
import React from "react";
import { localization } from "../Localization/localization";
import { WizardFooterStyles } from "./WizardFooter.styles";

export interface IWizardFooterProps {
    onNext: () => void;
    onPrevious?: () => void;
}

export class WizardFooter extends React.PureComponent<IWizardFooterProps> {
//     private static readonly classNames = mergeStyleSets({
//         frame: {
//             display: "inline-flex",
//             flexDirection: "row-reverse",
//             paddingTop: "10px",
//             paddingBottom: "10px"
//         },
//         next: {
//             height: "44px",
//             padding: "12px",
//             color: "#FFFFFF",
//             fontSize: "18px",
//             lineHeight: "24px",
//             backgroundColor: "#666666",
//             fontWeight: "400",
//             marginLeft: "10px"
//         },
//         back: {
//             height: "44px",
//             padding: "12px",
//             color: "#333333",
//             fontSize: "18px",
//             lineHeight: "24px",
//             backgroundColor: "#FFFFFF",
//             fontWeight: "400"
//         }
//     });
    
    public render(): React.ReactNode {
        const styles = WizardFooterStyles();
        return (
            <div className={styles.frame}>
                <PrimaryButton className={styles.next} text={localization.Footer.next} onClick={this.props.onNext}/>
                {(!!this.props.onPrevious) &&
                    <DefaultButton
                        className={styles.back}
                        text={localization.Footer.back}
                        onClick={this.props.onPrevious}/>
                }
            </div>
        );
    }
}