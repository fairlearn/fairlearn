import { ActionButton } from "office-ui-fabric-react/lib/Button";
import { Stack } from "office-ui-fabric-react/lib/Stack";
import { Text } from "office-ui-fabric-react";
import React from "react";
import { localization } from "../Localization/localization";
import { IntroTabStyles } from "./IntroTab.styles";

export interface IIntroTabProps {
    onNext: () => void;
}

export class IntroTab extends React.PureComponent <IIntroTabProps> {
    // private static readonly classNames = mergeStyleSets({
    //     firstSection: {
    //         padding: "43px 94px",
    //         fontFamily: `"Segoe UI", "Segoe UI Web (West European)", "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", sans-serif`,
    //         backgroundColor: "#333333",
    //         color: "#FFFFFF",
    //     },
    //     firstSectionTitle: {
    //         fontSize: "60px",
    //         lineHeight: "82px",
    //         fontWeight: "100"
    //     },
    //     firstSectionSubtitle: {
    //         fontSize: "60px",
    //         lineHeight: "82px",
    //         fontWeight: "500"
    //     },
    //     firstSectionBody: {
    //         paddingTop: "30px",
    //         paddingBottom: "70px",
    //         maxWidth: "500px",
    //         color: "#EBEBEB",
    //         fontWeight: "300",
    //         fontSize: "18px",
    //         lineHeight: "24px",
    //     },
    //     lowerSection: {
    //         padding: "50px 70px 90px 90px",
    //         backgroundColor: "#F2F2F2",
    //         color: "#333333",
    //         flexGrow: 1
    //     },
    //     stepsContainer: {
    //         display: "flex",
    //         flexDirection: "row",
    //         justifyContent: "space-between",
    //         borderBottom: "1px solid #CCCCCC",
    //         paddingBottom: "38px"
    //     },
    //     boldStep: {
    //         fontSize: "18px",
    //         lineHeight: "24px",
    //         fontWeight: "500",
    //         maxWidth: "300px",
    //         paddingRight: "25px",
    //         flex: 1,
    //     },
    //     numericLabel: {
    //         display:"inline-block",
    //         fontSize: "18px",
    //         lineHeight: "24px",
    //         fontWeight: "700",
    //         color: "#000000",
    //         width: "30px"
    //     },
    //     stepLabel: {
    //         color: "#333333",
    //         fontSize: "18px",
    //         lineHeight: "24px",
    //         fontWeight: "500",
    //     },
    //     explanatoryStep: {
    //         maxWidth: "300px",
    //         paddingRight: "20px",
    //         flex: 1
    //     },
    //     explanatoryText: {
    //         paddingTop: "15px",
    //         fontSize: "15px",
    //         lineHeight: "20px",
    //         color: "#666666"
    //     },
    //     getStarted: {
    //         paddingTop: "30px",
    //         color: "#333333",
    //         fontSize: "18px",
    //         lineHeight: "24px",
    //         fontWeight: "500"
    //     }
    // });

    render(): React.ReactNode {
        const styles = IntroTabStyles();
        return (<Stack style={{height: "100%"}}>
            <div className={styles.firstSection}>
                {/* <div className={styles.firstSectionTitle}>{localization.Intro.welcome}</div>
                <div className={styles.firstSectionSubtitle}>{localization.Intro.fairlearnDashboard}</div>
                <div className={styles.firstSectionBody}>{localization.Intro.introBody}</div> */}
                <Text className={styles.firstSectionTitle} block>{localization.Intro.welcome}</Text>
                <Text className={styles.firstSectionSubtitle} block>{localization.Intro.fairlearnDashboard}</Text>
                <Text variant={"large"} block>{localization.Intro.introBody}</Text>
            </div>
            <div className={styles.lowerSection}>
                <div className={styles.stepsContainer}>
                    {/* <div className={styles.boldStep}>{localization.Intro.explanatoryStep}</div> */}
                    <Text variant = {"large"} className={styles.boldStep}>{localization.Intro.explanatoryStep}</Text>
                    <div className={styles.explanatoryStep}>
                        <div>
                            {/* <span className={styles.numericLabel}>01</span> */}
                            <Text variant={"large"} className={styles.numericLabel}>01</Text>
                            {/* <span className={styles.stepLabel}>{localization.Intro.features}</span> */}
                            <Text variant={"large"}>{localization.Intro.features}</Text>
                        </div>
                        {/* <div className={styles.explanatoryText}>{localization.Intro.featuresInfo}</div> */}
                        <Text block>{localization.Intro.featuresInfo}</Text>
                    </div>
                    <div className={styles.explanatoryStep}>
                        <div>
                            {/* <span className={styles.numericLabel}>02</span> */}
                            <Text variant={"large"} className={styles.numericLabel}>02</Text>
                            {/* <span className={styles.stepLabel}>{localization.Intro.accuracy}</span> */}
                            <Text variant={"large"}>{localization.Intro.accuracy}</Text>
                        </div>
                        {/* <div className={styles.explanatoryText}>{localization.Intro.accuracyInfo}</div> */}
                        <Text block>{localization.Intro.accuracyInfo}</Text>
                    </div>
                </div>
                <Stack horizontalAlign={"center"}>
                    <ActionButton 
                        iconProps={{iconName: "Forward"}}
                        className={styles.getStarted}
                        onClick={this.props.onNext}>{localization.Intro.getStarted}</ActionButton>
                </Stack>
            </div>
        </Stack>);
    }
}