import React from "react";
import { Stack } from "office-ui-fabric-react/lib/Stack";
import { mergeStyleSets } from "@uifabric/styling";
import { Text } from "office-ui-fabric-react/lib/Text";
import { localization } from "../Localization/localization";
import { ActionButton } from "office-ui-fabric-react/lib/Button";

export interface IIntroTabProps {
    onNext: () => void;
}

export class IntroTab extends React.PureComponent <IIntroTabProps> {
    private static readonly classNames = mergeStyleSets({
        firstSection: {
            padding: "43px 94px",
            fontFamily: `"Segoe UI", "Segoe UI Web (West European)", "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", sans-serif`,
            backgroundColor: "#333333",
            color: "#FFFFFF",
        },
        firstSectionTitle: {
            fontSize: "60px",
            lineHeight: "82px",
            fontWeight: "100"
        },
        firstSectionSubtitle: {
            fontSize: "60px",
            lineHeight: "82px",
            fontWeight: "500"
        },
        firstSectionBody: {
            paddingTop: "30px",
            paddingBottom: "70px",
            maxWidth: "500px",
            color: "#EBEBEB",
            fontWeight: "300",
            fontSize: "18px",
            lineHeight: "24px",
        },
        lowerSection: {
            padding: "50px 94px 90px 94px",
            backgroundColor: "#F2F2F2",
            color: "#333333",
            flexGrow: 1
        },
        stepsContainer: {
            borderBottom: "1px solid #CCCCCC",
            paddingBottom: "38px"
        },
        boldStep: {
            fontSize: "18px",
            lineHeight: "24px",
            fontWeight: "500",
            width: "300px",
            paddingRight: "50px"
        },
        numericLabel: {
            display:"inline-block",
            fontSize: "18px",
            lineHeight: "24px",
            fontWeight: "700",
            color: "#000000",
            width: "30px"
        },
        stepLabel: {
            fontSize: "18px",
            lineHeight: "24px",
            fontWeight: "500",
        },
        explanatoryStep: {
            fontSize: "15px",
            lineHeight: "20px",
            color: "#666666",
            width: "300px"
        },
        getStarted: {
            paddingTop: "30px",
            fontSize: "30px"
        }
    });

    render(): React.ReactNode {
        return (<Stack style={{height: "100%"}}>
            <div className={IntroTab.classNames.firstSection}>
                <div className={IntroTab.classNames.firstSectionTitle}>{localization.Intro.welcome}</div>
                <div className={IntroTab.classNames.firstSectionSubtitle}>{localization.Header.title}</div>
                <div className={IntroTab.classNames.firstSectionBody}>{localization.loremIpsum}</div>
            </div>
            <div className={IntroTab.classNames.lowerSection}>
                <Stack wrap horizontal horizontalAlign={"space-between"} className={IntroTab.classNames.stepsContainer}>
                    <div className={IntroTab.classNames.boldStep}>{localization.loremIpsum}</div>
                    <div>
                        <div>
                            <span className={IntroTab.classNames.numericLabel}>01</span>
                            <span className={IntroTab.classNames.stepLabel}>Features</span>
                        </div>
                        <div className={IntroTab.classNames.explanatoryStep}>{localization.loremIpsum}</div>
                    </div>
                    <div>
                        <div>
                            <span className={IntroTab.classNames.numericLabel}>02</span>
                            <span className={IntroTab.classNames.stepLabel}>Accuracy</span>
                        </div>
                        <div className={IntroTab.classNames.explanatoryStep}>{localization.loremIpsum}</div>
                    </div>
                </Stack>
                <Stack horizontalAlign={"center"}>
                    <ActionButton 
                        iconProps={{iconName: "Forward"}}
                        className={IntroTab.classNames.getStarted}
                        onClick={this.props.onNext}>{localization.Intro.getStarted}</ActionButton>
                </Stack>
            </div>
        </Stack>);
    }
}